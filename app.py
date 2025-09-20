import streamlit as st
import numpy as np
import cv2
from PIL import Image
import requests
import io
import os
import gc

# Configure Streamlit with memory optimization
st.set_page_config(
    page_title="Depth Pro Portrait Mode",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory monitoring
def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    except:
        return 0

# Lightweight fallback processor (no AI model)
class FallbackPortraitProcessor:
    def __init__(self):
        self.name = "Fallback Processor"
    
    def estimate_depth(self, image):
        """Create depth map using edge detection and blur"""
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Use edge detection to find subject boundaries
        edges = cv2.Canny(img_array, 50, 150)
        
        # Distance from edges = likely depth
        distance_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Smooth and normalize
        depth = cv2.GaussianBlur(distance_transform, (15, 15), 0)
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        
        return depth.astype(np.float32)
    
    def auto_detect_subject(self, depth_map):
        """Find center-weighted subject"""
        h, w = depth_map.shape
        center_region = depth_map[h//4:3*h//4, w//4:3*w//4]
        return float(np.percentile(center_region, 75))
    
    def apply_portrait_blur(self, image, depth_map, focus_distance, blur_strength, focus_range):
        """Memory-efficient blur"""
        img_array = np.array(image, dtype=np.uint8)
        
        # Create focus mask
        distance_from_focus = np.abs(depth_map - focus_distance)
        focus_weights = np.exp(-(distance_from_focus ** 2) / (focus_range ** 2))
        
        # Smooth focus mask
        focus_weights = cv2.GaussianBlur(focus_weights, (11, 11), 3)
        
        # Apply single blur level (memory efficient)
        kernel_size = min(int(blur_strength * 8), 15) | 1  # Small kernel
        blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), kernel_size/3)
        
        # Blend
        blur_mask = (1 - focus_weights) * (blur_strength / 4.0)
        blur_mask = np.expand_dims(np.clip(blur_mask, 0, 1), axis=2)
        
        result = (1 - blur_mask) * img_array + blur_mask * blurred
        return np.clip(result, 0, 255).astype(np.uint8)

# Attempt to load AI model with memory management
class AIPortraitProcessor:
    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded = False
        
    def try_load_model(self):
        """Try to load AI model if memory allows"""
        initial_memory = get_memory_usage()
        
        if initial_memory > 500:  # If already using >500MB, skip AI
            return False
            
        try:
            import torch
            from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
            
            # Load with minimal memory footprint
            self.processor = DepthProImageProcessorFast.from_pretrained(
                "apple/DepthPro-hf"
            )
            
            self.model = DepthProForDepthEstimation.from_pretrained(
                "apple/DepthPro-hf",
                dtype=torch.float16,  # Half precision to save memory
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            self.loaded = True
            
            current_memory = get_memory_usage()
            if current_memory > 900:  # Close to 1GB limit
                self.unload_model()
                return False
                
            return True
            
        except Exception as e:
            st.warning(f"⚠️  AI model loading failed: {e}")
            self.unload_model()
            return False
    
    def unload_model(self):
        """Free up memory"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
        self.model = None
        self.processor = None
        self.loaded = False
        
        # Force garbage collection
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def estimate_depth(self, image):
        """AI depth estimation with fallback"""
        if not self.loaded:
            return None
            
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()
            
            # Immediate cleanup
            del inputs, outputs
            gc.collect()
            
            return depth.astype(np.float32)
            
        except Exception as e:
            st.error(f"❌ AI processing failed: {e}")
            self.unload_model()  # Free memory
            return None

def main():
    st.markdown("# 📸 Smart Portrait Mode")
    st.markdown("### Professional background blur with intelligent subject detection")
    
    # Memory status
    memory_usage = get_memory_usage()
    if memory_usage > 0:
        st.sidebar.info(f"💾 Memory: {memory_usage:.0f} MB")
    
    # Initialize processors
    fallback_processor = FallbackPortraitProcessor()
    ai_processor = None
    
    # Controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Processing mode
        use_ai = st.checkbox("🤖 Try AI Depth (if memory allows)", value=True)
        
        if use_ai and ai_processor is None:
            with st.spinner("🧠 Attempting to load AI model..."):
                ai_processor = AIPortraitProcessor()
                ai_loaded = ai_processor.try_load_model()
                
                if ai_loaded:
                    st.success("✅ AI model loaded successfully!")
                else:
                    st.warning("⚠️  Using fallback processing (memory limited)")
                    ai_processor = None
        
        # Settings
        auto_subject = st.checkbox("🎯 Auto-detect Subject", value=True)
        focus_distance = st.slider("Focus Distance", 0.0, 1.0, 0.6, 0.05)
        blur_strength = st.slider("Blur Strength", 0.0, 4.0, 2.0, 0.1)
        focus_range = st.slider("Focus Range", 0.1, 0.5, 0.2, 0.05)
        
        # Memory management
        if st.button("🔄 Free Memory"):
            if ai_processor:
                ai_processor.unload_model()
                ai_processor = None
            gc.collect()
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
        
        # URL option
        url_input = st.text_input("Or paste image URL:")
        if st.button("Load URL") and url_input:
            try:
                response = requests.get(url_input, timeout=10)
                uploaded_file = io.BytesIO(response.content)
                uploaded_file.name = "url_image.jpg"
                st.success("✅ Loaded from URL!")
            except:
                st.error("❌ Failed to load URL")
    
    # Process image
    if uploaded_file:
        try:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Aggressive resize for memory
            max_size = 400  # Very small to save memory
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Lanczos)
                st.info(f"📐 Resized to {image.size} for memory efficiency")
            
            with col1:
                st.subheader("📷 Original")
                st.image(image, width='stretch')  # Updated parameter
            
            # Process
            with st.spinner("🎨 Processing portrait effect..."):
                # Try AI first, fallback to basic
                depth_map = None
                processor_used = "Fallback"
                
                if ai_processor and ai_processor.loaded:
                    try:
                        depth_map = ai_processor.estimate_depth(image)
                        if depth_map is not None:
                            processor_used = "AI"
                    except:
                        pass
                
                # Use fallback if AI failed
                if depth_map is None:
                    depth_map = fallback_processor.estimate_depth(image)
                    processor = fallback_processor
                else:
                    processor = ai_processor
                
                # Auto-detect subject
                if auto_subject:
                    detected_focus = processor.auto_detect_subject(depth_map)
                    focus_distance = detected_focus
                
                # Apply blur
                result = processor.apply_portrait_blur(
                    image, depth_map, focus_distance, blur_strength, focus_range
                )
            
            with col2:
                st.subheader("✨ Portrait Mode")
                st.image(result, width='stretch')  # Updated parameter
                
                # Download
                result_pil = Image.fromarray(result)
                img_buffer = io.BytesIO()
                result_pil.save(img_buffer, format='PNG', quality=85)
                
                st.download_button(
                    "💾 Download Result",
                    data=img_buffer.getvalue(),
                    file_name=f"portrait_{processor_used.lower()}_{blur_strength:.1f}.png",
                    mime="image/png"
                )
            
            # Status
            st.success(f"✅ Processed with {processor_used} | Focus: {focus_distance:.2f} | Blur: {blur_strength:.1f}")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            
    else:
        with col2:
            st.info("👆 Upload an image to start!")
            st.markdown("**Features:**")
            st.markdown("• Smart subject detection")
            st.markdown("• Professional blur effects")
            st.markdown("• Works with limited memory")
            st.markdown("• AI enhancement when possible")

if __name__ == "__main__":
    main()
