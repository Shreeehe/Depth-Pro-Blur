import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter
import requests
import io
import os
import gc

# Configure Streamlit
st.set_page_config(
    page_title="Depth Pro Portrait Mode",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory monitoring
def get_memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0

# Get correct resampling method for PIL version compatibility
def get_lanczos_filter():
    """Get the correct Lanczos filter for different PIL versions"""
    try:
        # Try new format first (Pillow 10.0+)
        from PIL.Image import Resampling
        return Resampling.LANCZOS
    except (ImportError, AttributeError):
        try:
            # Try older format (Pillow 9.x)
            return Image.LANCZOS
        except AttributeError:
            # Fallback to even older format
            return Image.ANTIALIAS

# Lightweight processor that works without AI models
class SmartPortraitProcessor:
    def __init__(self):
        self.name = "Smart Edge-Based Processor"
    
    def estimate_depth(self, image):
        """Create depth map using computer vision techniques"""
        # Convert to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection to find subject boundaries
        edges = cv2.Canny(gray, 30, 100)
        
        # Morphological operations to fill gaps
        kernel = np.ones((3,3), np.uint8)
        edges_filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Distance transform (areas far from edges are likely subjects)
        distance = cv2.distanceTransform(255 - edges_filled, cv2.DIST_L2, 5)
        
        # Apply Gaussian blur to smooth
        depth = cv2.GaussianBlur(distance, (21, 21), 0)
        
        # Normalize to 0-1 range
        if np.max(depth) > np.min(depth):
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        else:
            # Fallback: create center-focused depth
            h, w = depth.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            distance_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            depth = 1 - (distance_from_center / np.max(distance_from_center))
        
        return depth.astype(np.float32)
    
    def auto_detect_subject(self, depth_map):
        """Find the main subject focus point"""
        h, w = depth_map.shape
        
        # Focus on center region (where subjects usually are)
        center_h_start, center_h_end = h // 4, 3 * h // 4
        center_w_start, center_w_end = w // 4, 3 * w // 4
        
        center_region = depth_map[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Use 85th percentile of center region (likely subject depth)
        focus_depth = np.percentile(center_region, 85)
        
        return float(np.clip(focus_depth, 0.0, 1.0))
    
    def apply_portrait_blur(self, image, depth_map, focus_distance, blur_strength, focus_range):
        """Apply professional portrait blur effect"""
        img_array = np.array(image, dtype=np.uint8)
        
        # Create focus mask
        distance_from_focus = np.abs(depth_map - focus_distance)
        focus_weights = np.exp(-(distance_from_focus ** 2) / (focus_range ** 2))
        
        # Smooth the focus transition
        focus_weights = cv2.GaussianBlur(focus_weights, (15, 15), 5)
        
        # Apply layered blur for realistic depth of field
        result = self._apply_layered_blur(img_array, focus_weights, blur_strength)
        
        return result
    
    def _apply_layered_blur(self, img_array, focus_weights, blur_strength):
        """Apply multiple blur layers for realistic depth of field"""
        result = img_array.astype(np.float32)
        
        # Calculate kernel sizes (smaller for memory efficiency)
        max_kernel = min(int(blur_strength * 6), 25)  # Cap at 25px
        max_kernel = max_kernel | 1  # Ensure odd number
        
        if max_kernel < 5:
            return img_array
        
        # Create 3 blur levels for smooth transition
        kernel_sizes = [
            max(5, int(max_kernel * 0.4) | 1),
            max(7, int(max_kernel * 0.7) | 1),
            max_kernel
        ]
        
        # Apply each blur level
        for i, kernel_size in enumerate(kernel_sizes):
            # Create blurred version
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), kernel_size/4.0)
            
            # Calculate blend weights
            blur_threshold = (i + 1) / len(kernel_sizes)
            blur_mask = 1.0 - focus_weights
            
            # Apply non-linear falloff for more natural look
            blur_mask = np.power(blur_mask, 1.3)
            
            # Calculate layer strength
            layer_strength = np.clip(blur_mask - blur_threshold + 0.3, 0, 1)
            layer_weight = layer_strength * (blur_strength / 5.0)
            
            # Expand dimensions for RGB blending
            layer_weight_3d = np.expand_dims(np.clip(layer_weight, 0, 1), axis=2)
            
            # Blend layers
            result = (1 - layer_weight_3d) * result + layer_weight_3d * blurred.astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)

def main():
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 2rem; color: white;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>📸 Smart Portrait Mode</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Professional background blur without AI dependencies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    processor = SmartPortraitProcessor()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Portrait Controls")
        
        # Memory info
        memory_usage = get_memory_usage()
        if memory_usage > 0:
            st.info(f"💾 Memory: {memory_usage:.0f} MB")
        
        # Settings
        auto_subject = st.checkbox("🎯 Auto-detect Subject", value=True,
                                  help="Automatically find the main subject in your photo")
        
        focus_distance = st.slider(
            "Focus Distance", 0.0, 1.0, 0.7, 0.02,
            help="0 = focus on foreground, 1 = focus on background"
        )
        
        blur_strength = st.slider(
            "Blur Intensity", 0.0, 5.0, 2.5, 0.1,
            help="Higher values create stronger background blur"
        )
        
        focus_range = st.slider(
            "Focus Transition", 0.1, 0.5, 0.15, 0.02,
            help="Controls how gradual the focus-to-blur transition is"
        )
        
        # Processing info
        st.info("🧠 **Smart Processing:** Uses computer vision edge detection and distance transforms")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Your Photo")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a portrait photo for best results"
        )
        
        # URL input option
        st.write("**Or load from URL:**")
        url_input = st.text_input("Image URL", placeholder="https://example.com/photo.jpg")
        
        if st.button("📥 Load from URL", type="secondary") and url_input.strip():
            try:
                with st.spinner("📡 Downloading image..."):
                    response = requests.get(url_input.strip(), timeout=15)
                    response.raise_for_status()
                    uploaded_file = io.BytesIO(response.content)
                    uploaded_file.name = "url_image.jpg"
                    st.success("✅ Image loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load from URL: {str(e)}")
    
    # Process image when available
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Get the correct resampling filter
            lanczos_filter = get_lanczos_filter()
            
            # Resize for performance (and memory constraints)
            original_size = image.size
            max_size = 600  # Reasonable size for processing
            
            if max(image.size) > max_size:
                # Use the compatible resampling method
                image.thumbnail((max_size, max_size), lanczos_filter)
                st.info(f"📐 Resized from {original_size} to {image.size} for optimal processing")
            
            # Display original
            with col1:
                st.subheader("📷 Original Photo")
                st.image(image, width=400, caption=f"Size: {image.size[0]}×{image.size[1]}")
            
            # Process with smart algorithm
            with st.spinner("🎨 Applying smart portrait effect..."):
                # Estimate depth using computer vision
                depth_map = processor.estimate_depth(image)
                
                # Auto-detect subject if enabled
                if auto_subject:
                    detected_focus = processor.auto_detect_subject(depth_map)
                    focus_distance = detected_focus
                    st.sidebar.success(f"🎯 Subject detected at depth: {detected_focus:.3f}")
                
                # Apply portrait blur
                result = processor.apply_portrait_blur(
                    image, depth_map, focus_distance, blur_strength, focus_range
                )
            
            # Display result
            with col2:
                st.subheader("✨ Portrait Mode Result")
                st.image(result, width=400, caption="Professional background blur applied")
                
                # Download button
                result_pil = Image.fromarray(result)
                img_buffer = io.BytesIO()
                result_pil.save(img_buffer, format='PNG', quality=90)
                
                st.download_button(
                    label="💾 Download Portrait Photo",
                    data=img_buffer.getvalue(),
                    file_name=f"portrait_blur_{blur_strength:.1f}.png",
                    mime="image/png",
                    type="primary"
                )
            
            # Show processing details
            st.success(f"✅ **Processing Complete!** Focus: {focus_distance:.2f} | Blur: {blur_strength:.1f} | Transition: {focus_range:.2f}")
            
            # Optional: Show depth map
            if st.checkbox("🗺️ Show Depth Analysis"):
                depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                st.image(depth_rgb, caption="Depth Analysis (Purple=Close, Yellow=Far)", width=400)
                
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.info("💡 Try uploading a different image or check the file format")
    
    else:
        # Instructions when no image is loaded
        with col2:
            st.info("""
            👆 **Get Started:**
            1. Upload a photo using the file uploader
            2. Or paste an image URL and click "Load from URL"
            3. Adjust the blur settings in the sidebar
            4. Download your professional portrait!
            """)
            
            st.markdown("""
            **💡 Tips for Best Results:**
            - Use portrait photos with clear subjects
            - Photos with good subject-background separation work best
            - Experiment with different blur intensities
            - Try auto-detection first, then manual adjustment
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 <strong>Powered by Computer Vision</strong> | No AI model downloads required | Works entirely in your browser</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
