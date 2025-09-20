import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import io
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Depth Pro Portrait Mode",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import torch and transformers
try:
    import torch
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    st.error(f"❌ PyTorch/Transformers not available: {e}")
    st.stop()

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

class StreamlitDepthProProcessor:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"
        
    @st.cache_resource
    def load_model(_self):
        """Load model with fixed warnings"""
        try:
            progress_bar = st.progress(0, "Loading Depth Pro model...")
            
            # Load processor
            progress_bar.progress(25, "Loading image processor...")
            _self.processor = DepthProImageProcessorFast.from_pretrained(
                "apple/DepthPro-hf",
                cache_dir="./model_cache",
                local_files_only=False
            )
            
            # Load model with fixed dtype parameter
            progress_bar.progress(75, "Loading depth estimation model...")
            _self.model = DepthProForDepthEstimation.from_pretrained(
                "apple/DepthPro-hf",
                dtype=torch.float32,  # Fixed: was torch_dtype
                cache_dir="./model_cache",
                local_files_only=False
            ).to("cpu")
            
            _self.model.eval()
            
            progress_bar.progress(100, "Model loaded successfully!")
            progress_bar.empty()
            
            return _self.processor, _self.model
            
        except Exception as e:
            st.error(f"❌ Failed to load Depth Pro model: {str(e)}")
            return None, None
    
    def estimate_depth(self, image):
        """Estimate depth with fallback"""
        try:
            if self.processor is None or self.model is None:
                self.processor, self.model = self.load_model()
                
            if self.processor is None:
                return self._create_simple_depth(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                post_processed = self.processor.post_process_depth_estimation(
                    outputs, target_sizes=[(image.height, image.width)]
                )
            
            depth = post_processed[0]["predicted_depth"].squeeze().cpu().numpy()
            
            # Clean up
            del inputs, outputs, post_processed
            
            # Validate depth
            depth = depth.astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
            
            return depth
            
        except Exception as e:
            st.warning(f"⚠️  Using fallback depth estimation: {str(e)}")
            return self._create_simple_depth(image)
    
    def _create_simple_depth(self, image):
        """Create simple depth map as fallback"""
        h, w = image.height, image.width
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        depth = 1 - (distance / np.max(distance))
        return depth.astype(np.float32)
    
    def auto_detect_subject(self, depth_map):
        """Auto-detect subject focus point"""
        try:
            h, w = depth_map.shape
            center_h, center_w = h // 2, w // 2
            region_size = min(h, w) // 4
            
            h_start = max(0, center_h - region_size)
            h_end = min(h, center_h + region_size)
            w_start = max(0, center_w - region_size)
            w_end = min(w, center_w + region_size)
            
            center_region = depth_map[h_start:h_end, w_start:w_end]
            focus_depth = np.percentile(center_region, 80)
            
            return float(np.clip(focus_depth, 0.0, 1.0))
        except:
            return 0.6
    
    def apply_portrait_blur(self, image, depth_map, focus_distance, blur_strength, focus_range):
        """Apply portrait blur effect"""
        try:
            image_array = np.array(image, dtype=np.uint8)
            
            # Normalize depth
            depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            
            # Create focus mask
            distance_from_focus = np.abs(depth_norm - focus_distance)
            focus_weights = np.exp(-(distance_from_focus ** 2) / (focus_range ** 2))
            focus_weights = cv2.GaussianBlur(focus_weights, (21, 21), 7)
            
            # Apply blur
            result = self._apply_variable_blur(image_array, focus_weights, blur_strength)
            return result
            
        except Exception as e:
            st.error(f"❌ Blur processing failed: {e}")
            return np.array(image)
    
    def _apply_variable_blur(self, image_array, focus_weights, blur_strength):
        """Apply variable blur"""
        try:
            result = image_array.astype(np.float32)
            
            max_kernel = min(int(blur_strength * 10), 21)
            max_kernel = max_kernel | 1
            
            if max_kernel < 5:
                return image_array
            
            blurred = cv2.GaussianBlur(image_array, (max_kernel, max_kernel), max_kernel/3.0)
            
            blur_mask = 1.0 - focus_weights
            blur_mask = np.power(blur_mask, 1.5)
            blur_mask = blur_mask * (blur_strength / 3.0)
            blur_mask_3d = np.expand_dims(np.clip(blur_mask, 0, 1), axis=2)
            
            result = (1 - blur_mask_3d) * result + blur_mask_3d * blurred.astype(np.float32)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            st.error(f"❌ Blur application failed: {e}")
            return image_array

def main():
    # Header
    st.markdown('<h1 class="main-header">📸 Depth Pro Portrait Mode</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered DSLR-style background blur")
    
    @st.cache_resource
    def get_processor():
        return StreamlitDepthProProcessor()
    
    processor = get_processor()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        auto_subject = st.checkbox("🎯 Auto-detect Subject", value=True)
        focus_distance = st.slider("Focus Distance", 0.0, 1.0, 0.6, 0.05)
        blur_strength = st.slider("Blur Strength", 0.0, 3.0, 1.5, 0.1)
        focus_range = st.slider("Focus Transition", 0.1, 0.5, 0.2, 0.05)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        url_input = st.text_input("Or enter image URL:")
        
        if st.button("Load from URL") and url_input.strip():
            try:
                with st.spinner("Downloading image..."):
                    response = requests.get(url_input.strip(), timeout=10)
                    response.raise_for_status()
                    uploaded_file = io.BytesIO(response.content)
                    uploaded_file.name = "url_image.jpg"
                    st.success("✅ Image loaded from URL!")
            except Exception as e:
                st.error(f"❌ Failed to load from URL: {e}")
    
    # Process image
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Resize for memory efficiency
            max_size = 600
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Lanczos)
                st.info(f"📐 Resized to {image.size} for faster processing")
            
            with col1:
                st.subheader("📷 Original")
                st.image(image, use_container_width=True)  # Fixed: was use_column_width
            
            # Process image
            with st.spinner("🎨 Processing with AI depth estimation..."):
                try:
                    depth_map = processor.estimate_depth(image)
                    
                    if auto_subject:
                        detected_focus = processor.auto_detect_subject(depth_map)
                        focus_distance = detected_focus
                        st.sidebar.success(f"🎯 Subject detected at depth: {detected_focus:.3f}")
                    
                    result = processor.apply_portrait_blur(
                        image, depth_map, focus_distance, blur_strength, focus_range
                    )
                    
                    with col2:
                        st.subheader("✨ Portrait Mode")
                        st.image(result, use_container_width=True)  # Fixed: was use_column_width
                        
                        # Download button
                        result_pil = Image.fromarray(result)
                        img_buffer = io.BytesIO()
                        result_pil.save(img_buffer, format='PNG', quality=90)
                        
                        st.download_button(
                            label="💾 Download Result",
                            data=img_buffer.getvalue(),
                            file_name=f"portrait_blur_{blur_strength:.1f}.png",
                            mime="image/png"
                        )
                    
                    st.success(f"✅ Applied: Focus={focus_distance:.2f} | Blur={blur_strength:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Image loading failed: {str(e)}")
    
    else:
        with col2:
            st.info("👆 Upload an image to get started!")

if __name__ == "__main__":
    main()
