import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import requests
import io
import os
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# Configure Streamlit page
st.set_page_config(
    page_title="Depth Pro Portrait Mode",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class OptimizedDepthProProcessor:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"  # Force CPU for Streamlit Cloud
        
    @st.cache_resource
    def load_model(_self):
        """Load model with caching and memory optimization"""
        try:
            with st.spinner("🚀 Loading Depth Pro model (first time may take 2-3 minutes)..."):
                # Load with CPU and memory optimization
                _self.processor = DepthProImageProcessorFast.from_pretrained(
                    "apple/DepthPro-hf",
                    cache_dir="./model_cache"  # Local cache
                )
                _self.model = DepthProForDepthEstimation.from_pretrained(
                    "apple/DepthPro-hf",
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    cache_dir="./model_cache"
                ).to("cpu")
                
                # Enable eval mode for inference
                _self.model.eval()
                
                return _self.processor, _self.model
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None, None
    
    def estimate_depth(self, image):
        """Memory-optimized depth estimation"""
        try:
            if self.processor is None or self.model is None:
                self.processor, self.model = self.load_model()
                
            if self.processor is None:
                return self._create_fallback_depth(image)
            
            # Process with memory management
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                post_processed = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[(image.height, image.width)]
                )
            
            depth = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()
            
            # Clean up GPU memory (if any)
            del inputs, outputs, post_processed
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Validate and clean depth
            depth = depth.astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
            
            return depth
            
        except Exception as e:
            st.warning(f"⚠️  Depth estimation issue: {e}")
            return self._create_fallback_depth(image)
    
    def _create_fallback_depth(self, image):
        """Create simple depth map if model fails"""
        h, w = image.height, image.width
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        return (distance / np.max(distance)).astype(np.float32)
    
    def auto_detect_subject(self, depth_map):
        """Detect main subject in image"""
        try:
            depth_clean = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
            depth_min, depth_max = np.min(depth_clean), np.max(depth_clean)
            
            if depth_max == depth_min:
                return 0.3
                
            depth_norm = (depth_clean - depth_min) / (depth_max - depth_min)
            
            # Focus on center region
            h, w = depth_norm.shape
            h_start, h_end = int(h * 0.25), int(h * 0.75)
            w_start, w_end = int(w * 0.25), int(w * 0.75)
            
            center_region = depth_norm[h_start:h_end, w_start:w_end]
            focus_depth = np.percentile(center_region, 20) if center_region.size > 0 else 0.3
            
            return float(np.clip(focus_depth, 0.0, 1.0))
            
        except:
            return 0.3
    
    def apply_portrait_blur(self, image, depth_map, focus_distance, blur_strength, focus_range):
        """Apply optimized portrait blur"""
        try:
            image_array = np.array(image, dtype=np.uint8)
            
            # Normalize depth
            depth_clean = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
            depth_min, depth_max = np.min(depth_clean), np.max(depth_clean)
            
            if depth_max == depth_min:
                h, w = depth_clean.shape
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h // 2, w // 2
                distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                depth_norm = distance / np.max(distance)
            else:
                depth_norm = (depth_clean - depth_min) / (depth_max - depth_min)
            
            # Create focus mask
            distance_from_focus = np.abs(depth_norm - focus_distance)
            focus_weights = np.exp(-(distance_from_focus ** 2) / (focus_range + 0.01))
            
            # Smooth focus mask
            focus_weights = cv2.GaussianBlur(focus_weights.astype(np.float32), (15, 15), 5)
            
            # Apply blur
            result = self._apply_blur_layers(image_array, focus_weights, blur_strength)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Blur error: {e}")
            return np.array(image)
    
    def _apply_blur_layers(self, image_array, focus_weights, blur_strength):
        """Memory-efficient blur application"""
        result = image_array.astype(np.float32)
        
        # Conservative kernel sizing for Streamlit Cloud
        h, w = image_array.shape[:2]
        max_kernel = min(int(blur_strength * 15), 31)  # Cap at 31 for memory
        max_kernel = max_kernel | 1  # Ensure odd
        
        # Only 2 blur levels to save memory
        kernel_sizes = [
            max(7, int(max_kernel * 0.5) | 1),
            max_kernel
        ]
        
        for i, kernel_size in enumerate(kernel_sizes):
            try:
                blurred = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), kernel_size/6.0)
                
                blur_threshold = (i + 1) / len(kernel_sizes)
                blend_mask = 1.0 - focus_weights
                layer_weight = np.clip(blend_mask - blur_threshold + 0.3, 0, 1)
                layer_weight = layer_weight * (blur_strength / 6.0)
                
                layer_weight_3d = np.expand_dims(np.clip(layer_weight, 0, 1), axis=2)
                result = (1 - layer_weight_3d) * result + layer_weight_3d * blurred.astype(np.float32)
                
            except Exception as e:
                st.warning(f"⚠️  Blur layer {i} failed: {e}")
                continue
        
        return np.clip(result, 0, 255).astype(np.uint8)

# Initialize processor
@st.cache_resource
def get_processor():
    return OptimizedDepthProProcessor()

def main():
    # Header
    st.markdown('<h1 class="main-header">📸 Depth Pro Portrait Mode</h1>', unsafe_allow_html=True)
    st.markdown("### Create professional DSLR-style background blur using AI depth estimation")
    
    # Initialize processor
    processor = get_processor()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        auto_subject = st.checkbox("🎯 Auto-detect Subject", value=True)
        
        focus_distance = st.slider(
            "Focus Distance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.02,
            help="0 = foreground focus, 1 = background focus"
        )
        
        blur_strength = st.slider(
            "Blur Strength", 
            min_value=0.0, 
            max_value=3.0, 
            value=1.5, 
            step=0.1,
            help="Higher values create stronger background blur"
        )
        
        focus_range = st.slider(
            "Focus Range", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2, 
            step=0.02,
            help="Controls how gradual the focus transition is"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo to apply portrait mode effect"
        )
        
        # URL input as alternative
        st.write("**Or enter image URL:**")
        url_input = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        
        if st.button("Load from URL") and url_input:
            try:
                response = requests.get(url_input, timeout=10)
                uploaded_file = io.BytesIO(response.content)
                uploaded_file.name = "url_image.jpg"
            except:
                st.error("❌ Failed to load image from URL")
    
    # Process image
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Resize for memory efficiency
            max_size = 800  # Smaller for Streamlit Cloud
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Lanczos)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_column_width=True)
            
            # Process with progress
            with st.spinner("🎨 Applying portrait effect..."):
                # Estimate depth
                depth_map = processor.estimate_depth(image)
                
                # Auto-detect subject if enabled
                if auto_subject:
                    detected_focus = processor.auto_detect_subject(depth_map)
                    focus_distance = detected_focus
                
                # Apply portrait blur
                result = processor.apply_portrait_blur(
                    image, depth_map, focus_distance, blur_strength, focus_range
                )
            
            with col2:
                st.subheader("✨ Portrait Mode Result")
                st.image(result, use_column_width=True)
                
                # Download button
                result_pil = Image.fromarray(result)
                img_buffer = io.BytesIO()
                result_pil.save(img_buffer, format='PNG', quality=95)
                
                st.download_button(
                    label="💾 Download Result",
                    data=img_buffer.getvalue(),
                    file_name=f"portrait_blur_{blur_strength:.1f}.png",
                    mime="image/png"
                )
            
            # Settings info
            st.info(f"🎯 **Settings Applied:** Focus: {focus_distance:.3f} | Blur: {blur_strength:.1f} | Range: {focus_range:.2f}")
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.info("💡 Try uploading a smaller image or refresh the page")
    
    else:
        with col2:
            st.info("👆 Upload an image or enter a URL to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Use portrait photos for best results")
    st.markdown("- Adjust focus distance if auto-detection isn't perfect") 
    st.markdown("- Higher blur strength creates more dramatic effects")

if __name__ == "__main__":
    main()
