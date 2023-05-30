elif classifier_name == "Logistic Regression":
        """
            Parameters:
                deposit_weight: (Integer) weight to be given to the deposits to deal with unbalanced data 
                penalty: (string) type of norm for the penalty 
                random_state: (Integer) seed for random generator, useful to obtain reproducible results 
                
            For more information about the model visit 
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """
        _verbose_print("Logistic Regression selected")
        penalty = parameter_dic["penalty"].valueAsText
        deposit_weight = parameter_dic["deposit_weight"].value
        random_state = parameter_dic["random_state"].value
        if deposit_weight is None:
            _verbose_print("deposit_weight is None, balanced wighting will be used")
            class_weight = "balanced"
        else:
            class_weight = {1: float(deposit_weight), -1: (100-float(deposit_weight))}

        classifier = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=1, fit_intercept=True,
                                        intercept_scaling=1, class_weight=class_weight, random_state=random_state,
                                        solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
                                        warm_start=False, n_jobs=1)