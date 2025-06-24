import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from typing import Tuple, List
from PIL import Image

# --- Genetic Algorithm Core Class (from your notebook) ---
# This class is copied directly from your project code.
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
        self.fitness_history =
        self.best_threshold_history =

    def otsu_fitness(self, image: np.ndarray, threshold: int) -> float:
        hist = cv2.calcHist([image], , None, , ).flatten()
        hist = hist.astype(np.float64)
        total_pixels = image.size
        prob = hist / total_pixels
        
        w0 = np.sum(prob[:threshold])
        w1 = np.sum(prob[threshold:])
        
        if w0 == 0 or w1 == 0:
            return 0
            
        mu0 = np.sum(np.arange(threshold) * prob[:threshold]) / w0
        mu1 = np.sum(np.arange(threshold, 256) * prob[threshold:]) / w1
        
        between_class_variance = w0 * w1 * (mu0 - mu1)**2
        return between_class_variance

    def initialize_population(self) -> List[int]:
        return [random.randint(1, 254) for _ in range(self.pop_size)]

    def tournament_selection(self, population: List[int], fitness: List[float]) -> int:
        tournament_size = 3
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def crossover(self, parent1: int, parent2: int) -> Tuple[int, int]:
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = int(alpha * parent1 + (1 - alpha) * parent2)
            child2 = int(alpha * parent2 + (1 - alpha) * parent1)
            child1 = max(1, min(254, child1))
            child2 = max(1, min(254, child2))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual: int) -> int:
        if random.random() < self.mutation_rate:
            noise = random.randint(-15, 15)
            mutated = individual + noise
            return max(1, min(254, mutated))
        return individual

    def optimize(self, image: np.ndarray, progress_bar) -> Tuple[int, float]:
        self.fitness_history =
        self.best_threshold_history =
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

# --- Helper Functions (from your notebook) ---
def load_and_preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    image = np.array(image)
    
    height, width = image.shape
    if height > 512 or width > 512:
        scale = 512 / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, new_height))
        
    image = cv2.medianBlur(image, 3)
    return image

def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image > threshold).astype(np.uint8) * 255

def calculate_traditional_otsu(image: np.ndarray) -> int:
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold)

def create_comprehensive_analysis(image: np.ndarray, ga_threshold: int, traditional_threshold: int, ga_optimizer: GAOtsuThresholding):
    # This function is copied directly from your project code to maintain the visualization style.
    plt.style.use('default')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. Citra Original
    axes.imshow(image, cmap='gray')
    axes.set_title('Citra Original', fontweight='bold')
    axes.axis('off')

    # 2. Histogram dengan threshold lines
    hist = cv2.calcHist([image], , None, , ).flatten()
    axes.plot(hist, color='blue', linewidth=2)
    axes.axvline(ga_threshold, color='red', linestyle='--', linewidth=2, label=f'GA: {ga_threshold}')
    axes.axvline(traditional_threshold, color='green', linestyle='-', linewidth=2, label=f'Otsu: {traditional_threshold}')
    axes.set_title('Histogram dengan Threshold', fontweight='bold')
    axes.set_xlabel('Intensitas Pixel')
    axes.set_ylabel('Frekuensi')
    axes.legend()
    axes.grid(True, alpha=0.3)

    # 3. Hasil GA Thresholding
    ga_result = apply_threshold(image, ga_threshold)
    axes.imshow(ga_result, cmap='gray')
    axes.set_title(f'GA Threshold: {ga_threshold}', fontweight='bold')
    axes.axis('off')

    # 4. Hasil Traditional Otsu
    otsu_result = apply_threshold(image, traditional_threshold)
    axes.imshow(otsu_result, cmap='gray')
    axes.set_title(f'Traditional Otsu: {traditional_threshold}', fontweight='bold')
    axes.axis('off')

    # 5. Evolusi Fitness
    axes.plot(ga_optimizer.fitness_history, 'b', linewidth=2, marker='o', markersize=3)
    axes.set_title('Evolusi Fitness', fontweight='bold')
    axes.set_xlabel('Generasi')
    axes.set_ylabel('Fitness Score')
    axes.grid(True, alpha=0.3)

    # 6. Evolusi Threshold
    axes[1, 1].plot(ga_optimizer.best_threshold_history, 'r', linewidth=2, marker='s', markersize=3)
    axes[1, 1].axhline(traditional_threshold, color='green', linestyle='--', linewidth=2, label=f'Otsu: {traditional_threshold}')
    axes[1, 1].set_title('Evolusi Threshold', fontweight='bold')
    axes[1, 1].set_xlabel('Generasi')
    axes[1, 1].set_ylabel('Nilai Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 7. Perbandingan Fitness
    ga_fitness = ga_optimizer.otsu_fitness(image, ga_threshold)
    otsu_fitness = ga_optimizer.otsu_fitness(image, traditional_threshold)
    methods =
    fitness_scores = [ga_fitness, otsu_fitness]
    bars = axes.[1, 2]bar(methods, fitness_scores, color=['red', 'green'], alpha=0.7)
    axes.[1, 2]set_title('Perbandingan Fitness', fontweight='bold')
    axes.[1, 2]set_ylabel('Fitness Score')
    for bar, score in zip(bars, fitness_scores):
        axes.[1, 2]text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.5f}', ha='center', va='bottom', fontweight='bold')

    # 8. Statistik Segmentasi
    ga_white_pixels = np.sum(ga_result == 255)
    otsu_white_pixels = np.sum(otsu_result == 255)
    total_pixels = image.size
    stats_text = f"""HASIL ANALISIS:
    GA Method:
    - Threshold: {ga_threshold}
    - Fitness: {ga_fitness:.6f}
    - Pixel Putih: {ga_white_pixels:,} ({ga_white_pixels/total_pixels*100:.1f}%)

    Traditional Otsu:
    - Threshold: {traditional_threshold}
    - Fitness: {otsu_fitness:.6f}
    - Pixel Putih: {otsu_white_pixels:,} ({otsu_white_pixels/total_pixels*100:.1f}%)

    Perbedaan:
    - Threshold: {abs(ga_threshold - traditional_threshold)}
    - Fitness: {abs(ga_fitness - otsu_fitness):.6f}
    """
    axes.[1, 3]text(0.05, 0.95, stats_text, transform=axes.[1, 3]transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes.[1, 3]axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Analisis Genetic Algorithm vs Traditional Otsu Thresholding', fontsize=16, fontweight='bold')
    return fig

# --- Streamlit App Main Interface ---
st.set_page_config(layout="wide")
st.title("Genetic Algorithm for Otsu Thresholding Optimization by image")

st.write("Upload an MRI image of a brain to perform segmentation using a Genetic Algorithm-optimized Otsu threshold and compare it with the traditional method.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = load_and_preprocess_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded and Preprocessed Image', cmap='gray')

    with col2:
        if st.button('Run Analysis', use_container_width=True):
            with st.spinner('Optimizing with Genetic Algorithm... Please wait.'):
                # Initialize and run GA
                ga_optimizer = GAOtsuThresholding(pop_size=30, generations=50, mutation_rate=0.1, crossover_rate=0.8)
                
                progress_bar = st.progress(0, text="Starting Optimization...")
                ga_threshold, ga_fitness = ga_optimizer.optimize(image, progress_bar)
                
                # Calculate traditional Otsu for comparison
                traditional_threshold = calculate_traditional_otsu(image)
                
                st.success('Optimization Complete!')

            # Create and display the comprehensive analysis plot
            st.write("### Analysis Results")
            fig = create_comprehensive_analysis(image, ga_threshold, traditional_threshold, ga_optimizer)
            st.pyplot(fig)
