import argparse 
import logging
from data_utils.load_data import load_data
import pandas as pd
from models.models import MyEnsembleModel

def main():
    parser = argparse.ArgumentParser(description="Course Name: Machine Learning")
    logging.basicConfig(level=logging.INFO)

    parser.add_argument("--model", choices=["lgbm+dtr", "gbr+rfr", "lgbm+xg"], help="lgbm: LightGBM, dtr: Decision Tree Regressor, gbr: Gradient Boosting Regressor, rfr: Random Forest Regressor, xg: xgBoost", required=True)
    args = parser.parse_args()

    #load data
    logging.info("LOADING DATA...")
    df = pd.read_csv(r"D:\HK4\DS102-machine-learning\src_code\data\data.csv")
    data = load_data(df)

    #init model
    myModel = MyEnsembleModel(data["in"], data["out"])

    #train and evaluate model
    if args.model == "lgbm+dtr":
        logging.info("TRAINING LIGHTGBM + DECISION TREE REGRESSOR...")
        dtr_r2, dtr_mse, lgbm_r2, lgbm_mse, pred_r2, pred_mse = myModel.lgbm_dtr()
        print("""            RESULT___________________
            __________R2__________MSE
            DTR_:____{:.2f}_______{:.2f}
            LGBM:____{:.2f}_______{:.2f}
            PLUS:____{:.2f}_______{:.2f}
        """.format(dtr_r2, dtr_mse, lgbm_r2, lgbm_mse, pred_r2, pred_mse))

    elif args.model == "gbr+rfr":
        logging.info("TRAINING GRADIENT BOOSTING REGRESSOR + RANDOM FOREST REGRESSOR...")
        r2_score, mean_r2 = myModel.gbr_rfr()
        print("""            RESULT___________
            ______________R2_
            k=1:_________{:.2f}
            k=2:_________{:.2f}
            k=3:_________{:.2f}
            k=4:_________{:.2f}
            k=5:_________{:.2f}
            mean:________{:.2f}
        """.format(r2_score[0], r2_score[1], r2_score[2], r2_score[3], r2_score[4], mean_r2))

    elif args.model == "lgbm+xg":
        logging.info("TRAINING LIGHTGBM + XGBOOST...")
        r2_score, mean_r2 = myModel.lgbm_xg()
        print("""            RESULT___________
            ______________R2_
            k=1:_________{:.2f}
            k=2:_________{:.2f}
            k=3:_________{:.2f}
            k=4:_________{:.2f}
            k=5:_________{:.2f}
            mean:________{:.2f}
        """.format(r2_score[0], r2_score[1], r2_score[2], r2_score[3], r2_score[4], mean_r2))
    else:
        print("Invalid Task")
    
if __name__ == "__main__":
    main()
