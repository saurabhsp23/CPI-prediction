# Module: model_training.py

class Combo():
    def __init__(self, df=None, agg_func=None, filt_func=None, smooth_func=None, tester=None, lookback_period=None):
        
        self.df = df
        self.agg_func = agg_func
        self.filt_func = filt_func
        self.smooth_func = smooth_func
        self.test = tester
        self.lookback_period = lookback_period
        
        
        
    '''
    Processes the dataframe passed in with the aggregate function, filter function, then smooth function
    in that order. Can eventually update this to allow for multiple aggregate, filter, and smoothing operations
    or to have the order of operations changed.
    '''
    
    def process_df(self):
        try:
            if self.agg_func:
                self.df = self.agg_func(self.df)
        except:
            print('Aggregate function input/output error')
        
        try:
            if self.filt_func:
                self.df = self.filt_func(self.df)
        except:
            print('Filter function input/output error')
        
        try: 
            if self.smooth_func:
                self.df = self.smooth_func(self.df)
        except:
            print('Smooth function input/output error')
            
            
    def performance_metrics(self, y_pred, y_act):
        '''
        Print performance metrics such as R-squared, MAE, and MAPE.

        Args:
            y_pred (list): Predicted values.
            y_act (list): Actual values.
        '''
        rsquared = r2_score(y_pred,y_act)
        mae = mean_absolute_error(y_pred,y_act)
        mape = mean_absolute_percentage_error(y_pred,y_act)

        # calculating f stat
        var_oos = np.var(y_act)
        var_pred = np.var(y_pred)

        # Perform F-test
        F_statistic = var_pred / var_oos

        # Calculate degrees of freedom
        dof_pred = len(y_pred) - 1
        dof_oos = len(y_act) - 1

        # Calculate p-value
        p_value = f.sf(F_statistic, dof_pred, dof_oos)

        print(f'R-squared: {rsquared}, MAE: {mae}, MAPE: {mape}, p_value: {p_value}')
        
        return (rsquared, mae, mape, p_value)
    

    
    
    def run_model(self, model = LinearRegression(), label = 'LinearRegression'):
        '''
        Walk forward train/validate split using a machine learning model.

        Args:
            df (pd.DataFrame): Combined DataFrame with monthly aggregated data and indicator data.
            lookback_period (int): The number of periods to lo/ok back for training.
            model: The machine learning model to use for predictions.
        '''
        
        lookback_period = self.lookback_period
        df=self.df
        
        df = df.ffill()#.dropna()
        df = df.dropna(axis=1, how='all')  # drop cols with all Nans
        df=df.replace(np.nan, 0) # now dropping any intial rows with nans
        total_periods = len(df)
        y_pred = []  # predicted values
        y_act = []  # actual values
        indicator = df.columns[-1]
        reg_betas = pd.DataFrame(index=df.columns[:-1])

        for i in range(total_periods - lookback_period + 1):
            df_lookback = df.iloc[i:i + lookback_period + 1]

            # TODO IMPORTANT NOTE: think more carefully how we fill nans in the below instead of a
            # back-fill, forward-fill
            # df_lookback = df_lookback.ffill() # backfill nan's currently ... how to make better?
            # and forward fill the final row
            if len(df_lookback.columns) <= 1:
                continue

            df_train = df_lookback.head(-1)
            df_test = df_lookback.tail(1)
            y_act.append(float(df_test.iloc[:, -1].values))
            # train linear regression model and get predicted value
            xtrain = df_train.iloc[:, 0:-1]
            ytrain = df_train.iloc[:, -1]
            xtest = df_test.iloc[:, :-1]  # predictors

            if label == 'XGBoosting':
                # Use XGBoost specific code
                xg_train = xgb.DMatrix(xtrain.values, label=ytrain.values)
                xg_test = xgb.DMatrix(xtest.values)
                params = {
                    'tree_method': 'auto',  # Ensure tree_method is set to "auto"
                    'objective': 'reg:squarederror',  # Use regression objective
                }
                model = xgb.train(params, xg_train)
                # Update y_pred using XGBoost predictions
                y_pred.append(float(model.predict(xg_test)[0]))
            else:
                model.fit(xtrain.values, ytrain.values)
                # sm_lasso = sm.OLS(ytrain.values, xtrain.values)
                ypred = model.predict(xtest.values)
                y_pred.append(float(ypred))

                model_params = pd.DataFrame(model.coef_, index=list(xtrain.columns))
                reg_betas.loc[model_params.index, xtest.index] = model_params.values
        if label == 'XGBoosting':
                xgb.plot_importance(model)
                plt.show()
                importance = model.get_score(importance_type='weight')
                
        return self.performance_metrics(y_pred, y_act), reg_betas


    

class Agg():
    '''
    Class containing all the different aggregate functions. When adding new functions, just add a new static function
    to the class and give it a distinct name. All the input outputs of the functions must be of the same format
    unless in special cases, but then the user must be careful with handling the rest of the pipeline.
    '''

    TREAT_AS_MULTI = {96: [1109, 1110, 1111], 316: [1820, 1821, 1822, 1823], 597: [1808, 1809, 1810, 1811, 1812, 1813], 664: [1827, 1829, 1830, 1831, 1832, 1833]}
    GRADIENT = {97: [], 98: [], 101: [], 103: [], 104: [], 105: [], 106: [], 107: [], 108: [], 109: [], 110: [], 153: [], 154: [], 155: [], 156: [], 158: [], 159: [], 170: [], 173: [], 182: [], 183: [], 316: [1820, 1821, 1822, 1823, 1824], 335: [], 337: [], 338: [], 339: [], 340: [], 341: [], 342: [], 583: [], 584: [], 585: [], 586: [1090], 587: [1093], 595: [], 663: [], 664: [1827, 1829, 1830, 1831, 1832, 1833, 1834], 2400: [], 2401: [], 2402: [], 2403: [], 2406: [], 2428: [], 2429: [3370], 2464: [3671], 2465: [3677], 2466: [3683, 3684], 2467: [3690], 2468: [3695], 2474: [3742]}
    UNCERTAINTY = {585:1087, 586:1090, 587:1093, 2429:3370, 2464:3671, 2465:3677, 2466:3683, 2467:3690, 2468:3695, 2474:3742}
    SUBCAT = {'Economic Expectations': [2464, 2465, 2466, 2467, 2468],
                'Economic Trends': [338, 339, 340, 341, 342, 583, 584, 585, 586, 587],
                'Employment': [316,  664, 2406, 2407],
                'Employment Effects': [597],
                'General Shopping Trends': [182, 183],
                'Grocery': [595],
                'Home Improvement': [100, 663],
                'Home Ownership': [ 96, 101],
                'Home Value Prediction': [97, 98],
                'Inflation': [2426, 2427, 2428, 2429, 2474],
                'Online Retail': [170, 173],
                'Personal Finances': [335,  337, 2400, 2401, 2402, 2403],
                'Physical Retail': [153, 154, 155, 156, 158, 159, 184],
                'Spending Expectations': [103, 104, 105, 106, 107, 108, 109, 110]}
    
    
    def __init__(self, combinations):
        self.combinations = combinations
    
    '''
    run_agg_process: Takes in a list of lists that provides the names of the functions to aggregate. Inner lists
    are 'steps' and they do the listed aggregations in parallel and then append the resulting columns.
    Each subsequent step will then do new aggregations based on the dataframe produced by the previous step rather
    than on the original dataframe until the procedure is over.
    Example: [['agg1', 'agg2'], ['agg3', 'agg4']]
    In that example, agg1 and agg2 will be applied on the dataframe and those outputs will be concatenated
    horizontally. Then, agg3 and agg4 will be run on that resulting dataframe and those outputs will be concatenated
    horizontally again as the final output.
    
    Input: DataFrame to aggregate; list of lists that describes the procedure to aggregate
    
    Output: DataFrame of aggregated data
    '''
    @staticmethod
    def run_agg_process(df, agg_procedure):
        current_step = df.copy()
        for step in agg_procedure:
            next_step = pd.DataFrame(index=df.index)
            
            for sub_func in step:
                agg_func = getattr(Agg, sub_func)
                next_step = pd.concat([next_step, agg_func(current_step)], axis=1)
            
            current_step = next_step
            
        return current_step
    
    '''
    multi_and_gradient: Aggregate the question-answer pairs based on whether they are multi-select or not level. 
    Specifically, treat multi-select questions as default and aggregate by each non-multiselect quesion by assigning 
    value to correseponding answers. For Uncertainty feature, identify answers indicating uncertainty and aggregate 
    them using a mean into a time series.

    Input: DataFrame of 'daily_aggregate.pq'

    Output: DataFrame
    '''
    @staticmethod
    def multi_and_gradient(df, treat_as_multiselect=TREAT_AS_MULTI, gradient=GRADIENT, uncertainty=UNCERTAINTY):

        df_exp = pd.DataFrame(index=df.index)
        # 1. Treat "multiselect" quesitons as default
        if treat_as_multiselect:
            temp = []
            for qid, aid in treat_as_multiselect.items():
                pattern = r'^(' + '|'.join('Q' + str(qid) + 'A' + re.escape(str(id)) for id in aid) + r')'
                data = df.filter(regex=pattern)
                # Check whether this question is in df
                if data.columns.empty:
                    continue
                else: 
                    temp.append(data)
            df_exp = pd.concat(temp, axis=1)

        if gradient:
        # 2. Aggregate answers by each non-multiselect quesion
            # Select non-multiselect quesion
            for qid, aid in gradient.items():
                data = df.filter(regex='Q' + str(qid))
                
                # Check whether this question is in df
                if data.columns.empty:
                    continue
                
                # Drop the answers that are regard as "multi-select"
                if aid:
                    lst = ['Q' + str(qid) + 'A' + str(id) for id in aid]
                    data = data.drop(columns=lst)

                # Calculate weighted average value of each question, by assigning value 1, 2, 3, ... to each answer respectively
                length = len(data.columns) + 1
                order = np.array(list(range(1, length)))
                data = data @ order
            
                # Standardize
                data /= length
                
                df_exp['Q' + str(qid) + 'Non_Multi'] = data

        return df_exp.dropna(axis=1, how='all')
    

    '''
    Agg2: Aggregate the answers by subcategory level. 
    Filter the intra-question aggregated dataframe by each subcategory and perform basic aggregations, including mean and PCA(require non-Nan data)

    Input: DataFrame

    Output: DataFrame
    '''
    @staticmethod
    def agg2(df, subcat=SUBCAT, pca=False, correlation=False):
        df_sub_agg = pd.DataFrame(index = df.index)

        # Loop through each subcategory
        for qid, aid in subcat.items():
            # Filter out the data under some specific subcategory
            pattern = r'^(' + '|'.join('Q' + re.escape(str(id)) for id in aid) + r')'
            data = df.filter(regex = pattern)

            if correlation:
                print(qid)
                display(data.corr())

            if pca:
                # PCA model require non-NaN data
                if data.isna().any().any():
                    raise ValueError('There is NaN in the dataframe!')
                
                # Build the PCA model
                if len(data.columns) > 1:
                    pca_model = PCA()
                    pca_model.fit(data)
                    transformed_data = pca_model.transform(data)
                    # choose the component with the largest variance as the aggregation outcome of the subcategory
                    df_sub_agg[qid] = pd.DataFrame(transformed_data, index=df.index).iloc[:,0]

                else:
                    df_sub_agg[qid] = data
            else:
                # Take the mean as the aggregation outcome of the subcategory
                df_sub_agg[qid] = data.mean(axis=1)
                
        return df_sub_agg
    
    @staticmethod
    def uncertainty_agg_groups(df):
        return Agg.uncertainty_agg_helper(df)
    
    @staticmethod
    def uncertainty_agg_no_groups(df):
        return Agg.uncertainty_agg_helper(df, groups=False)
    
    @staticmethod
    def uncertainty_agg_helper(df, uncertainty=UNCERTAINTY, uncertainty_groups=UNCERTAINTY_GROUPS, groups=True):
        
        # This function does not drop the first row to keep formatting consistent for future functions,
        # but doing so would be beneficial since the first row doesn't seem to contain healthy data
        
        # Note that this function does not smooth, but this aggregation works best with a multi-day rolling average
        
        # Filter for only the uncertainty questions
        pattern = r'^(' + '|'.join('Q' + str(qid) + 'A' + re.escape(str(aid)) for qid, aid in uncertainty.items()) + r')'
        uncert = df.filter(regex=pattern)
        
        # Retrieve the expected percentages of the uncertain response (calculated by taking 1 / # of possible answers)
        ans_count = {}
        for c in df.columns:
            prefix = c.split('A')[0][1:]
            if int(prefix) in uncertainty.keys():
                ans_count[prefix] = ans_count.get(prefix, 0) + 1
        normal_percentage = {key: 1 / value for key, value in ans_count.items()}
        
        # Convert values to excess values by subtracting expected percentages
        for key, value in normal_percentage.items():
            cols = [col for col in uncert.columns if col.startswith('Q' + key)]
            for col in cols:
                uncert[col] -= value
        
        # Construct output dataframe
        output = pd.DataFrame(index=uncert.index)
        
        if not groups:
            output['excess mean'] = uncert.mean(axis=1)
            return output
        
        # This part is for if we want to split by subcategory groups
        uncert.columns = uncert.columns.str.extract(r'Q(\d+)A').squeeze()
        
        for key in uncertainty_groups:
            group = uncertainty_groups[key]
            group_df = uncert[group]
            output[key + ' excess mean'] = group_df.mean(axis=1)
            
        return output
    
    @staticmethod
    def subcat_agg(df, subcat=SUBCAT):
        output = pd.DataFrame(index=df.index)
        
        for cat, qs in subcat.items():
            regex_pattern = r'^Q(?:' + '|'.join(map(str, qs)) + r')[A-Za-z].*'
            cat_filt = df.filter(regex=regex_pattern, axis=1)
            df[cat] = cat_filt.mean(axis=1)
    
        regex_pattern = r'^Q\d+[A-Za-z].*'
        non_q_cols = df.columns[~df.columns.str.contains(regex_pattern)]
        non_q_df = df.loc[:, non_q_cols]
        
        return pd.concat([output, non_q_df], axis=1)

# Content from cell_06.py
# Content from cell 6

# Aggregate demo

df = pd.read_pickle('dadf_relevant.pkl').asfreq('d')
Agg.run_agg_process(df, [['multi_and_gradient', 'uncertainty_agg_groups'], ['subcat_agg']])

# Responsible for training, optimizing, and saving models.

def train_model(*args, **kwargs):
    """Placeholder for train_model"""
    pass

def optimize_model(*args, **kwargs):
    """Placeholder for optimize_model"""
    pass

def save_model(*args, **kwargs):
    """Placeholder for save_model"""
    pass

