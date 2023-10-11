if __name__ == '__main__': 

    import pickle
    import sys 
    sys.path.append("..")
    sys.path.append("../d2c/")

    with open('../data/ts.pkl', 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    #drop na 
    observations = [obs.dropna() for obs in observations]
    
    # with open('../data/dag.pkl', 'rb') as f:
    #     observations, dags = pickle.load(f)

    from d2c.D2C import D2C
    d2c = D2C(dags,observations,n_jobs=10)
    d2c.initialize()
    d2c.save_descriptors_df('../data/ts_descriptors_with_cycles.csv')
