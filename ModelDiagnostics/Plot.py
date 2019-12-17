#  imports

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.compat import lzip
import json
import numpy as np


class LinearRegressionResidualPlot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        linear_model = sm.OLS(self.y, sm.add_constant(self.x)).fit()

        return linear_model

    @staticmethod
    def check_linearity_assumption(fitted_y, residuals):
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(fitted_y, residuals,
                                       lowess=True,
                                       scatter_kws={'alpha': 0.5},
                                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        plot_1.axes[0].set_title('Residuals vs Fitted')
        plot_1.axes[0].set_xlabel('Fitted values')
        plot_1.axes[0].set_ylabel('Residuals')
        plt.savefig("ResVsFitted.png")

    @staticmethod
    def check_residual_normality(residuals_normalized):
        qq = ProbPlot(residuals_normalized)
        plot_2 = qq.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        plot_2.axes[0].set_title('Normal Q-Q')
        plot_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_2.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(residuals_normalized)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            plot_2.axes[0].annotate(i,
                                    xy=(np.flip(qq.theoretical_quantiles, 0)[r],
                                        residuals_normalized[i]))

        plt.savefig("Normality.png")

    @staticmethod
    def check_homoscedacticity(fitted_y, residuals_norm_abs_sqrt):
        plot_3 = plt.figure()
        plt.scatter(fitted_y, residuals_norm_abs_sqrt, alpha=0.5)
        sns.regplot(fitted_y, residuals_norm_abs_sqrt,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plot_3.axes[0].set_title('Scale-Location')
        plot_3.axes[0].set_xlabel('Fitted values')
        plot_3.axes[0].set_ylabel("$\\sqrt{|Standardized Residuals|}$")

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residuals_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            plot_3.axes[0].annotate(i,
                                    xy=(fitted_y[i],
                                        residuals_norm_abs_sqrt[i]))
        plt.savefig("Homoscadasticity.png")

    @staticmethod
    def check_influcence(leverage, cooks, residuals_normalized):
        plot_4 = plt.figure()
        plt.scatter(leverage, residuals_normalized, alpha=0.5)
        sns.regplot(leverage, residuals_normalized,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plot_4.axes[0].set_xlim(0, max(leverage) + 0.01)
        plot_4.axes[0].set_ylim(-3, 5)
        plot_4.axes[0].set_title('Residuals vs Leverage')
        plot_4.axes[0].set_xlabel('Leverage')
        plot_4.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
        for i in leverage_top_3:
            plot_4.axes[0].annotate(i,
                                    xy=(leverage[i],
                                        residuals_normalized[i]))
        plt.savefig("Influence.png")

    def diagnostic_plots(self, linear_model):
        """

        :param linear_model: Linear Model Fit on the Data
        :return: None

        This method validates the assumptions of Linear Model
        """
        diagnostic_result = {}

        summary = linear_model.summary()
        #diagnostic_result['summary'] = str(summary)

        # fitted values
        fitted_y = linear_model.fittedvalues
        # model residuals
        residuals = linear_model.resid

        # normalized residuals
        residuals_normalized = linear_model.get_influence().resid_studentized_internal

        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(residuals_normalized))

        # leverage, from statsmodels internals
        leverage = linear_model.get_influence().hat_matrix_diag

        # cook's distance, from statsmodels internals
        cooks = linear_model.get_influence().cooks_distance[0]

        self.check_linearity_assumption(fitted_y, residuals)

        self.check_residual_normality(residuals_normalized)

        self.check_homoscedacticity(fitted_y, model_norm_residuals_abs_sqrt)

        self.check_influcence(leverage, cooks, residuals_normalized)

        # 1. Non-Linearity Test
        try:
            name = ['F value', 'p value']
            test = sms.linear_harvey_collier(linear_model)
            linear_test_result = lzip(name, test)
        except Exception as e:
            linear_test_result = str(e)
        diagnostic_result['Non_Linearity_Test'] = linear_test_result

        # 2. Hetroskedasticity Test
        name = ['Lagrange multiplier statistic', 'p-value',
                'f-value', 'f p-value']
        test = sms.het_breuschpagan(linear_model.resid, linear_model.model.exog)
        test_val = lzip(name, test)
        diagnostic_result['Hetroskedasticity_Test'] = test_val

        # 3. Normality of Residuals
        name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
        test = sms.jarque_bera(linear_model.resid)
        test_val = lzip(name, test)
        diagnostic_result['Residual_Normality_Test'] = test_val

        # 4. MultiCollnearity Test
        test = np.linalg.cond(linear_model.model.exog)
        test_val = [('condition no',test)]
        diagnostic_result['MultiCollnearity_Test'] = test_val

        # 5. Residuals Auto-Correlation Tests
        test = sms.durbin_watson(linear_model.resid)
        test_val = [('p value', test)]
        diagnostic_result['Residual_AutoCorrelation_Test'] = test_val

        json_result = json.dumps(diagnostic_result)
        return summary, json_result
