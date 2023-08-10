if __name__ == '__main__': 

    import pickle
    import sys 
    sys.path.append("..")
    sys.path.append("../d2c/")

    with open('observations.pkl', 'rb') as f:
        observations = pickle.load(f)

    with open('dags.pkl', 'rb') as f:
        dags = pickle.load(f)

    with open('updated_dags.pkl', 'rb') as f:
        updated_dags = pickle.load(f)

    from d2c.D2C import D2C
    d2c = D2C(dags,observations,n_jobs=10)
    d2c.initialize()
    d2c.save_descriptors_df('descriptors.csv')
