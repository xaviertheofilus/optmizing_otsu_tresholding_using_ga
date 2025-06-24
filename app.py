import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List
from PIL import Image

# Error handling untuk OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV tidak dapat diimpor: {e}")
    st.info("Mencoba menggunakan alternatif untuk beberapa fungsi OpenCV...")
    CV2_AVAILABLE = False

# --- Genetic Algorithm Core Class ---
class GAOtsuThresholding:
    """
    Implementasi Genetic Algorithm untuk optimasi Otsu Thresholding
    dengan fokus pada segmentasi citra medis (tumor otak)
    """
    def __init__(self, pop_size=30, generations=50, mutation_rate=0.1, crossover_rate=0.8):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # FIXED: Initialize lists properly
        self.fitness_history = []
        self.best_threshold_history = []

    def otsu_fitness(self, image: np.ndarray, threshold: int) -> float:
        """Menghitung fungsi fitness berdasarkan kriteria Otsu."""
        if CV2_AVAILABLE:
            # Menggunakan OpenCV
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        else:
            # Alternatif tanpa OpenCV
            hist, _ = np.histogram(image, bins=256, range=(0, 256))
        
        hist = hist.astype(np.float64)
        total_pixels = image.size
        prob = hist / total_pixels
        
        w0 = np.sum(prob[:threshold])
        w1 = np.sum(prob[threshold:])
        
        if w0 == 0 or w1 == 0:
            return 0
            
        mu0 = np.sum(np.arange(threshold) * prob[:threshold]) / w0
        mu1 = np.sum(np.arange(threshold, 256) * prob[threshold:]) / w1
        
        between_class_variance = w0 * w1 * ((mu0 - mu1)**2)
        return between_class_variance

    def initialize_population(self) -> List[int]:
        """Membuat populasi awal dengan distribusi yang baik."""
        return [random.randint(1, 254) for _ in range(self.pop_size)]

    def tournament_selection(self, population: List[int], fitness: List[float]) -> int:
        """Seleksi turnamen untuk memilih parent terbaik."""
        tournament_size = 3
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        """Crossover aritmatik untuk menghasilkan offspring."""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = int(alpha * parent1 + (1 - alpha) * parent2)
            child2 = int(alpha * parent2 + (1 - alpha) * parent1)
            child1 = max(1, min(254, child1))
            child2 = max(1, min(254, child2))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual: int) -> int:
        """Mutasi dengan menambahkan noise acak."""
        if random.random() < self.mutation_rate:
            noise = random.randint(-15, 15)
            mutated = individual + noise
            return max(1, min(254, mutated))
        return individual

    def optimize(self, image: np.ndarray, progress_bar) -> Tuple[int, float]:
        """Proses optimasi utama GA."""
        self.fitness_history = []
        self.best_threshold_history = []
        population = self.initialize_population()

        for gen in range(self.generations):
            fitness_scores = [self.otsu_fitness(image, thresh) for thresh in population]
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_threshold = population[best_idx]
            
            self.fitness_history.append(best_fitness)
            self.best_threshold_history.append(best_threshold)
            
            progress_bar.progress((gen + 1) / self.generations, text=f"Generation {gen+1}/{self.generations}")

            new_population = [best_threshold]  # Elitism
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]

        final_fitness = [self.otsu_fitness(image, thresh) for thresh in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx], final_fitness[best_idx]

# --- Helper Functions ---
def load_and_preprocess_image(uploaded_file) -> np.ndarray:
    """Memuat, mengubah ke grayscale, resize, dan mengurangi noise pada citra."""
    image = Image.open(uploaded_file).convert('L')
    image = np.array(image)
    
    height, width = image.shape
    if height > 512 or width > 512:
        scale = 512 / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        # Menggunakan PIL untuk resize jika OpenCV tidak tersedia
        if CV2_AVAILABLE:
            image = cv2.resize(image, (new_width, new_height))
        else:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image = np.array(pil_image)
    
    # Median blur menggunakan scipy jika OpenCV tidak tersedia
    if CV2_AVAILABLE:
        image = cv2.medianBlur(image, 3)
    else:
        from scipy import ndimage
        image = ndimage.median_filter(image, size=3)
    
    return image

def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Aplikasi thresholding biner."""
    return (image > threshold).astype(np.uint8) * 255

def calculate_traditional_otsu(image: np.ndarray) -> int:
    """Hitung threshold menggunakan metode Otsu tradisional."""
    if CV2_AVAILABLE:
        threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(threshold)
    else:
        # Implementasi Otsu manual
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total_pixels = image.size
        prob = hist / total_pixels
        
        max_variance = 0
        best_threshold = 0
        
        for threshold in range(1, 255):
            w0 = np.sum(prob[:threshold])
            w1 = np.sum(prob[threshold:])
            
            if w0 == 0 or w1 == 0:
                continue
                
            mu0 = np.sum(np.arange(threshold) * prob[:threshold]) / w0
            mu1 = np.sum(np.arange(threshold, 256) * prob[threshold:]) / w1
            
            between_class_variance = w0 * w1 * ((mu0 - mu1)**2)
            
            if between_class_variance > max_variance:
                max_variance = between_class_variance
                best_threshold = threshold
        
        return best_threshold

def create_comprehensive_analysis(image: np.ndarray, ga_threshold: int, traditional_threshold: int, ga_optimizer: GAOtsuThresholding):
    """Membuat visualisasi komprehensif untuk analisis hasil."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # FIXED: Use proper indexing for all subplots
    # 1. Citra Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    # 2. Histogram dengan threshold lines
    if CV2_AVAILABLE:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    else:
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    axes[0, 1].plot(hist, color='blue', linewidth=2)
    axes[0, 1].axvline(ga_threshold, color='red', linestyle='--', linewidth=2, label=f'GA: {ga_threshold}')
    axes[0, 1].axvline(traditional_threshold, color='green', linestyle='-', linewidth=2, label=f'Otsu: {traditional_threshold}')
    axes[0, 1].set_title('Histogram with Thresholds', fontweight='bold')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Hasil GA Thresholding
    ga_result = apply_threshold(image, ga_threshold)
    axes[0, 2].imshow(ga_result, cmap='gray')
    axes[0, 2].set_title(f'GA Threshold: {ga_threshold}', fontweight='bold')
    axes[0, 2].axis('off')

    # 4. Hasil Traditional Otsu
    otsu_result = apply_threshold(image, traditional_threshold)
    axes[0, 3].imshow(otsu_result, cmap='gray')
    axes[0, 3].set_title(f'Traditional Otsu: {traditional_threshold}', fontweight='bold')
    axes[0, 3].axis('off')

    # 5. Evolusi Fitness
    axes[1, 0].plot(ga_optimizer.fitness_history, 'b', linewidth=2, marker='o', markersize=3)
    axes[1, 0].set_title('Fitness Evolution', fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Fitness Score')
    axes[1, 0].grid(True, alpha=0.3)

    # 6. Evolusi Threshold
    axes[1, 1].plot(ga_optimizer.best_threshold_history, 'r', linewidth=2, marker='s', markersize=3)
    axes[1, 1].axhline(traditional_threshold, color='green', linestyle='--', linewidth=2, label=f'Otsu: {traditional_threshold}')
    axes[1, 1].set_title('Threshold Evolution', fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Threshold Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 7. Perbandingan Fitness
    ga_fitness = ga_optimizer.otsu_fitness(image, ga_threshold)
    otsu_fitness = ga_optimizer.otsu_fitness(image, traditional_threshold)
    # FIXED: Initialize list with values
    methods = ['GA Method', 'Traditional Otsu']
    fitness_scores = [ga_fitness, otsu_fitness]
    # FIXED: Fix syntax for bar plot
    bars = axes[1, 2].bar(methods, fitness_scores, color=['red', 'green'], alpha=0.7)
    axes[1, 2].set_title('Fitness Comparison', fontweight='bold')
    axes[1, 2].set_ylabel('Fitness Score')
    for bar, score in zip(bars, fitness_scores):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.5f}', ha='center', va='bottom', fontweight='bold')

    # 8. Statistik Segmentasi
    ga_white_pixels = np.sum(ga_result == 255)
    otsu_white_pixels = np.sum(otsu_result == 255)
    total_pixels = image.size
    stats_text = f"""ANALYSIS RESULTS:
GA Method:
- Threshold: {ga_threshold}
- Fitness: {ga_fitness:.6f}
- White Pixels: {ga_white_pixels:,} ({ga_white_pixels/total_pixels*100:.1f}%)

Traditional Otsu:
- Threshold: {traditional_threshold}
- Fitness: {otsu_fitness:.6f}
- White Pixels: {otsu_white_pixels:,} ({otsu_white_pixels/total_pixels*100:.1f}%)

Difference:
- Threshold: {abs(ga_threshold - traditional_threshold)}
- Fitness: {abs(ga_fitness - otsu_fitness):.6f}
"""
    # FIXED: Fix syntax for text box
    axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Genetic Algorithm vs Traditional Otsu Analysis', fontsize=16, fontweight='bold')
    return fig

# --- Streamlit App Main Interface ---
st.set_page_config(layout="wide")
st.title("Genetic Algorithm for Otsu Thresholding Optimization")

st.write("Upload an MRI image of a brain to perform segmentation using a Genetic Algorithm-optimized Otsu threshold and compare it with the traditional method.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = load_and_preprocess_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded and Preprocessed Image', use_column_width=True)

    with col2:
        if st.button('Run Analysis', use_container_width=True):
            with st.spinner('Optimizing with Genetic Algorithm... Please wait.'):
                ga_optimizer = GAOtsuThresholding(pop_size=30, generations=50, mutation_rate=0.1, crossover_rate=0.8)
                
                progress_bar = st.progress(0, text="Starting Optimization...")
                ga_threshold, _ = ga_optimizer.optimize(image, progress_bar)
                
                traditional_threshold = calculate_traditional_otsu(image)
                
                st.success('Optimization Complete!')

            st.write("### Analysis Results")
            fig = create_comprehensive_analysis(image, ga_threshold, traditional_threshold, ga_optimizer)
            st.pyplot(fig)
