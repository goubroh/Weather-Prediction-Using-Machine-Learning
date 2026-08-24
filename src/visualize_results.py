import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_methodology_block_diagram(output_path="methodology_block_diagram.png"):
    # Set publication-quality figure canvas (IEEE format wide aspect ratio)
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Academic Muted Color Palette
    PRIMARY = "#1E3A8A"      # Navy Blue
    SECONDARY = "#0284C7"    # Steel Blue
    ACCENT = "#0D9488"       # Teal
    CONTAINER = "#F8FAFC"    # Cool Light Grey
    BORDER = "#334155"       # Slate Border
    TEXT_DARK = "#0F172A"    # Dark Slate
    SUCCESS = "#15803D"      # Muted Green

    def draw_section_container(x, y, w, h, title):
        """Draws a dashed background container for grouped pipeline stages."""
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            linewidth=1.2, edgecolor=BORDER, facecolor=CONTAINER, linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 2.5, title, ha='left', va='center', 
                fontsize=9.5, fontweight='bold', color=PRIMARY)

    def draw_box(x, y, w, h, title, body="", bg_color="#FFFFFF", border_color=BORDER, text_color=TEXT_DARK):
        """Draws a structured block with proper inner text padding to avoid overlap."""
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            linewidth=1.1, edgecolor=border_color, facecolor=bg_color, alpha=0.98
        )
        ax.add_patch(rect)
        
        if body:
            # Title anchored near top
            ax.text(x + w/2, y + h - 2.8, title, ha='center', va='center', 
                    fontsize=8.5, fontweight='bold', color=text_color)
            # Body text centered in remaining box space
            ax.text(x + w/2, y + (h - 2.8)/2, body, ha='center', va='center', 
                    fontsize=7.5, color='#334155', linespacing=1.3)
        else:
            # Single centered text label
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=8.5, fontweight='bold', color=text_color)

    def draw_arrow(x1, y1, x2, y2):
        """Draws clean connecting flow arrows."""
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.3, mutation_scale=10)
        )

    # =========================================================
    # STAGE 1: DATA ACQUISITION & PREPROCESSING
    # =========================================================
    draw_section_container(3, 76, 94, 19, "Phase I: Data Acquisition & Target Construction")

    draw_box(5, 78, 25, 12, "Seattle Weather Dataset", "Daily Obs: Temp Max/Min,\nPrecipitation, Wind Speed", bg_color="#EFF6FF", border_color=PRIMARY)
    draw_arrow(30, 84, 36, 84)

    draw_box(36, 78, 25, 12, "Target Variable", "Binary Label Binarization:\nRain = 1, Non-Rain = 0", bg_color="#E0F2FE", border_color=PRIMARY)
    draw_arrow(61, 84, 67, 84)

    draw_box(67, 78, 28, 12, "Leak-Proof Split", "80:20 Stratified Partition\nChronological Sorting First", bg_color="#BAE6FD", border_color=PRIMARY)

    # Connector down to Phase II
    draw_arrow(81, 78, 81, 69)

    # =========================================================
    # STAGE 2: DOMAIN-SPECIFIC FEATURE ENGINEERING PIPELINE
    # =========================================================
    draw_section_container(3, 37, 94, 32, "Phase II: Domain-Specific Feature Engineering Pipeline (Replaces Variance PCA)")

    # Feature Sub-blocks
    draw_box(6, 42, 19, 21, "1. Cyclic Encoding", "Preserves Seasonal\nContinuity:\n\n• Sin/Cos (Month)\n• Sin/Cos (Day of Year)", bg_color="#F0FDFA", border_color=ACCENT)
    draw_arrow(25, 52.5, 28, 52.5)

    draw_box(28, 42, 19, 21, "2. Thermal Indices", "Temperature Dynamics:\n\n• Daily Temp Range\n  (Max - Min)\n• Average Temp", bg_color="#CCFBF1", border_color=ACCENT)
    draw_arrow(47, 52.5, 50, 52.5)

    draw_box(50, 42, 19, 21, "3. Multi-Day Lags", "Temporal History\nAutocorrelation:\n\n• 1, 2, & 3-Day Lags\n  (Precip, Temp, Wind)", bg_color="#99F6E4", border_color=ACCENT)
    draw_arrow(69, 52.5, 72, 52.5)

    draw_box(72, 42, 22, 21, "4. Rolling Window", "Short-Term Trends:\n\n• 3-Day Rolling Avg\n• StandardScaler\n  (Fit on Train Partition)", bg_color="#5EEAD4", border_color=ACCENT)

    # Connector down to Phase III
    draw_arrow(83, 42, 83, 31)

    # =========================================================
    # STAGE 3: MODEL BENCHMARKING & EVALUATION
    # =========================================================
    draw_section_container(3, 4, 94, 27, "Phase III: Machine Learning Framework & Comparative Evaluation")

    draw_box(6, 8, 20, 18, "Base Classifiers", "• Decision Tree (Depth=6)\n• Support Vector Machine\n• Logistic Regression\n• Naïve Bayes | KNN", bg_color="#FEF3C7", border_color="#D97706")
    draw_arrow(26, 17, 28, 17)

    draw_box(28, 8, 20, 18, "Tree Ensembles", "• Random Forest\n• Gradient Boosting\n• Hist Gradient Boosting", bg_color="#FDE68A", border_color="#D97706")
    draw_arrow(48, 17, 50, 17)

    draw_box(50, 8, 20, 18, "Soft-Voting Meta-Ensemble", "Probability Average:\n• Random Forest +\n• Gradient Boosting +\n• Support Vector Machine", bg_color="#F59E0B", border_color="#B45309", text_color="#FFFFFF")
    draw_arrow(70, 17, 73, 17)

    draw_box(73, 7, 22, 20, "Evaluation Metrics", "Binary Outputs & Metrics:\n\n• Accuracy: 96.23%\n• ROC-AUC: 0.9824\n• Precision: 98.35%\n• Recall: 92.97%\n• F1-Score: 95.58%", bg_color="#DCFCE7", border_color=SUCCESS)

    # Figure Caption
    plt.suptitle("Fig. 1. Methodological Workflow of the Proposed Domain-Engineered Weather Prediction Framework",
                 y=0.98, fontsize=11, fontweight='bold', color=TEXT_DARK)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    generate_methodology_block_diagram()