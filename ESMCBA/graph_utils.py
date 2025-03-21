# graph_utils.py

# We import everything from imports.py to ensure consistency:
from ESMCBA.imports import *

def plot_predictions(predictions_df, predicted_model, title_prefix=''):
    """
    Create a 2x2 subplot figure with:
      - Scatter plot with density coloring
      - Scatter plot + hexbin with marginal histograms
      - Residuals vs Actual
      - KDE plot (Actual vs Predicted)
    Also display Spearman, Pearson, MSE, MAE, R^2 in the figure subtitle.
    """

    # --- Compute metrics ---
    spearman_rho, _ = spearmanr(predictions_df, predicted_model)
    pearson_r, _    = pearsonr(predictions_df, predicted_model)
    mse             = mean_squared_error(predictions_df, predicted_model)
    mae             = mean_absolute_error(predictions_df, predicted_model)
    r2              = r2_score(predictions_df, predicted_model)
    
    # Create an overall title that includes the summary metrics (as a subtitle)
    overall_title   = title_prefix if title_prefix else "Model Predictions"
    subtitle        = (
        f"Spearman Rho: {spearman_rho:.2f} | Pearson R: {pearson_r:.2f} | "
        f"MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}"
    )
    
    # Convert to numpy arrays
    x = np.array(predictions_df)
    y = np.array(predicted_model)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    
    # Create a 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(6, 8))
    
    #
    # Panel A: Scatter plot with density coloring and identity line
    #
    ax = axs[0, 0]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()  # sort points by density so densest are plotted on top
    x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]
    scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=10, cmap='viridis')
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Identity Line')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Actual Log Binding Affinity (nM)')
    ax.set_ylabel('Predicted Log Binding Affinity (nM)')
    ax.set_title('Scatter Plot with Density')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    #
    # Panel B: Scatter plot + hexbin with marginal histograms
    #
    ax = axs[0, 1]
    divider = make_axes_locatable(ax)
    ax_marg_x = divider.append_axes("top", size="25%", pad=0.1, sharex=ax)
    ax_marg_y = divider.append_axes("right", size="25%", pad=0.1, sharey=ax)
    
    # Hide the tick labels on the marginal axes
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Force a square (equal aspect) display
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')

    hb = ax.hexbin(x, y, gridsize=30, mincnt=1)
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Identity Line')
    ax.set_xlabel('Actual Log Binding Affinity (nM)')
    ax.set_ylabel('Predicted Log Binding Affinity (nM)')

    # Marginal distributions
    ax_marg_x.hist(x, bins=20, alpha=0.7)
    ax_marg_y.hist(y, bins=20, alpha=0.7, orientation='horizontal')

    #
    # Panel C: Residuals vs. Actual plot
    #
    ax = axs[1, 0]
    residuals = x - y
    ax.scatter(x, residuals, c='blue', alpha=0.7, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Actual Log Binding Affinity (nM)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs. Actual')

    #
    # Panel D: KDE plot of Actual vs. Predicted distributions
    #
    ax = axs[1, 1]
    sns.kdeplot(x, shade=True, label='Actual', ax=ax)
    sns.kdeplot(y, shade=True, label='Predicted', ax=ax)
    ax.set_xlabel('Log Binding Affinity (nM)')
    ax.set_title('Density Plot of Actual vs. Predicted')
    ax.legend()

    # Final figure title
    fig.suptitle(f"{overall_title}\n{subtitle}", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
