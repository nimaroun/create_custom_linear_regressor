from sklearn.datasets._samples_generator import make_regression


def create_sample_regression_data(df):

    """
    Create sample regression dataset with 1 feature and length 200
    """
    X, y = (make_regression(n_samples = 200, 
                            n_features = 1, 
                            n_informative = 1, 
                            noise=6,
                            bias=30,
                            random_state=200)
            )

    return X,y

