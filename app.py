import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator - Gabriel Velazquez Berrueta",
    page_icon="🔢GVB",
    layout="wide"
)

# Set device
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# VAE Model Definition (same as training script)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        inputs = torch.cat([z, y], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    return F.one_hot(labels, num_classes=num_classes).float()

@st.cache_resource
def load_model():
    """Load the trained VAE model"""
    try:
        # Initialize model
        model = VAE(input_dim=784, latent_dim=20, hidden_dim=400, num_classes=10)
        
        # Load state dict (you'll need to upload the trained model file)
        checkpoint = torch.load('mnist_vae_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error("Model file 'mnist_vae_model.pth' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_digit_samples(model, digit, num_samples=5, latent_dim=20):
    """Generate samples for a specific digit"""
    if model is None:
        return None
        
    model.eval()
    with torch.no_grad():
        # Create one-hot encoded label for the digit
        label = torch.tensor([digit] * num_samples).to(device)
        label_onehot = one_hot_encode(label)
        
        # Sample from latent space with some randomness
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate samples
        samples = model.decode(z, label_onehot)
        
        # Reshape to 28x28 images
        samples = samples.view(num_samples, 28, 28)
        
    return samples.cpu().numpy()

def create_mnist_style_display(images, digit):
    """Create a display similar to MNIST dataset format"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(f'Generated Digit: {digit}', fontsize=16, fontweight='bold')
    
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')
        
        # Add border to make it look more like MNIST
        for spine in axes[i].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str

# Streamlit App
def main():
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("---")
    
    st.markdown("""
    This application generates handwritten digit images similar to the MNIST dataset using a trained Variational Autoencoder (VAE).
    
    **Instructions:**
    1. Select a digit (0-9) from the dropdown below
    2. Click "Generate Images" to create 5 unique samples
    3. The generated images will be displayed in MNIST-style format
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure the trained model file is available.")
        st.markdown("""
        **To use this app:**
        1. First run the training script to create `mnist_vae_model.pth`
        2. Make sure the model file is in the same directory as this app
        3. Refresh the page
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        # Digit selection
        selected_digit = st.selectbox(
            "Select digit to generate:",
            options=list(range(10)),
            index=0,
            help="Choose which digit (0-9) you want to generate"
        )
        
        # Generate button
        generate_button = st.button(
            "🎲 Generate Images",
            type="primary",
            help="Click to generate 5 new images of the selected digit"
        )
        
        # Add some info
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.markdown("- Architecture: Conditional VAE")
        st.markdown("- Dataset: MNIST")
        st.markdown("- Image size: 28×28 pixels")
        st.markdown("- Generates: 5 unique samples")
    
    with col2:
        st.subheader("Generated Images")
        
        if generate_button:
            with st.spinner(f"Generating images for digit {selected_digit}..."):
                # Generate images
                generated_images = generate_digit_samples(model, selected_digit, num_samples=5)
                
                if generated_images is not None:
                    # Create MNIST-style display
                    fig = create_mnist_style_display(generated_images, selected_digit)
                    
                    # Display the images
                    st.pyplot(fig)
                    
                    # Add download option
                    img_str = fig_to_base64(fig)
                    st.markdown(
                        f'<a href="data:image/png;base64,{img_str}" download="digit_{selected_digit}_samples.png">📥 Download Images</a>',
                        unsafe_allow_html=True
                    )
                    
                    # Show individual images in a grid
                    st.markdown("### Individual Images")
                    cols = st.columns(5)
                    for i, img in enumerate(generated_images):
                        with cols[i]:
                            # Convert to PIL Image for display
                            img_pil = Image.fromarray((img * 255).astype(np.uint8))
                            st.image(img_pil, caption=f"Sample {i+1}", use_container_width=True)
                
                else:
                    st.error("Failed to generate images. Please try again.")
        
        else:
            st.info("👆 Select a digit and click 'Generate Images' to see the results!")
            
            # Show example placeholder
            st.markdown("### Example Output")
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
                    caption="Example MNIST digits (this is what the generated images will look like)", 
                    width=400)

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:**
    - Built with Streamlit and PyTorch
    - Uses a Conditional Variational Autoencoder (CVAE) trained on MNIST dataset
    - Generates diverse handwritten digit samples
    - Each generation produces unique variations of the selected digit
    """)

if __name__ == "__main__":
    main()