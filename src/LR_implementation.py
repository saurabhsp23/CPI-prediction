

class InvalidOperation(Exception):
    pass

    LOOKBACK_PERIOD = 70
    DATA_BASE_PATH = '../Data'
    # Includes all testing models here for later use
    MODELS = []

# Content from cell_03.py
# Content from cell 3

class StepwiseRegression:
    def __init__(self, criterion='aic', verbose=False):
        self.criterion = criterion
        self.verbose = verbose
        self.features = []
        self.model = LinearRegression()

    def fit(self, X, y, stop_criterion=None):
        n = X.shape[0]
        p = X.shape[1]
        self.features = []  # To store the selected features

        if stop_criterion is None:
            stop_criterion = p

        if self.criterion == 'aic':
            best_criterion = np.inf
        else:
            best_criterion = -np.inf

        for _ in range(stop_criterion):
            remaining_features = [f for f in range(p) if f not in self.features]
            if len(remaining_features) == 0:
                break

            best_feature = None
            best_model = None

            for feature in remaining_features:
                selected_features = self.features + [feature]
                X_selected = X[:, selected_features]
                X_selected = sm.add_constant(X_selected)
                model = sm.OLS(y, X_selected).fit()

                if self.criterion == 'aic':
                    current_criterion = model.aic
                    if current_criterion < best_criterion:
                        best_criterion = current_criterion
                        best_feature = feature
                        best_model = model
                else:  # Default to BIC
                    current_criterion = model.bic
                    if current_criterion > best_criterion:
                        best_criterion = current_criterion
                        best_feature = feature
                        best_model = model

            if best_feature is not None:
                if self.verbose:
                    print(f"Adding feature {best_feature} to the model")
                self.features.append(best_feature)
                self.model = best_model
            else:
                break

    def predict(self, X):
        X_selected = X[:, self.features]
        X_selected_with_const = sm.add_constant(X_selected, has_constant='add')
        return self.model.predict(X_selected_with_const)

