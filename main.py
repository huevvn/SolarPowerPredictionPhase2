import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import preprocessing as prep
import analysis as ana
import models as mdl
import evaluation as evl

warnings.filterwarnings('ignore')

DATA_PATH = 'data/AswanData_weatherdata.csv'
OUT_DIR = 'output'
PLOTS_DIR = f'{OUT_DIR}/visualization'
CSVS_DIR = f'{OUT_DIR}/csv_output'

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSVS_DIR, exist_ok=True)
    
    # ===== LOAD AND CLEAN =====
    print('\n-> LOADING DATA')
    df = prep.load_data(DATA_PATH)
    print(f'Raw rows: {len(df)}')
    
    df = prep.handle_missing(df)
    df = prep.remove_outliers(df)
    print(f'After cleaning: {len(df)} rows')
    
    # ===== PREPROCESSING & EDA =====
    print('\n-> EXPLORATORY DATA ANALYSIS')
    # descriptive stats
    stats = ana.descriptive_stats(df)
    print('\nDescriptive Statistics:')
    print(stats)
    stats.to_csv(f'{CSVS_DIR}/descriptive_stats.csv')
    
    # correlation and covariance
    corr = ana.correlation_matrix(df)
    cov = ana.covariance_matrix(df)
    corr.to_csv(f'{CSVS_DIR}/correlation_matrix.csv')
    cov.to_csv(f'{CSVS_DIR}/covariance_matrix.csv')
    print('\nCorrelation matrix saved')
    print('Covariance matrix saved')
    
    # heatmap
    ana.plot_heatmap(corr, f'{PLOTS_DIR}/correlation_heatmap.png')
    
    # distribution plots for each feature
    print('\nGenerating distribution plots...')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col != 'Unnamed: 0':
            ana.plot_dist(df, col, f'{PLOTS_DIR}/dist_{col}.png')
    
    # boxplots
    target = 'SolarPV'
    df_model = df.drop(columns=['Date', 'Unnamed: 0'], errors='ignore')
    df_cls = prep.bin_target(df_model, target)
    
    for col in ['Temperature', 'Humidity', 'DewPoint']:
        if col in df_cls.columns:
            ana.plot_boxplot(df_cls, target+'_Class', col, f'{PLOTS_DIR}/box_{col}.png')
    
    # ===== STATISTICAL TESTS =====
    print('\n-> STATISTICAL TESTS')

    # anova test
    f_stat, p_val = ana.anova_test(df_cls, target+'_Class', 'Temperature')
    print(f'\nANOVA (Temperature vs SolarPV_Class): F={f_stat:.3f}, p={p_val:.4f}')
    if p_val < 0.05:
        print('  => Significant difference between groups')
    
    # t-test between high and low humidity
    high_hum = df_cls[df_cls['Humidity'] > df_cls['Humidity'].median()][target]
    low_hum = df_cls[df_cls['Humidity'] <= df_cls['Humidity'].median()][target]
    t_stat, t_p = ana.ttest(high_hum, low_hum)
    print(f'\nT-Test (High vs Low Humidity on Solar): t={t_stat:.3f}, p={t_p:.4f}')
    
    # chi-square test
    df_cls['Temp_Cat'] = pd.qcut(df_cls['Temperature'], q=3, labels=['Cold', 'Mild', 'Hot'])
    chi, chi_p = ana.chi_square(df_cls, 'Temp_Cat', target+'_Class')
    print(f'\nChi-Square (Temp_Cat vs SolarPV_Class): chi2={chi:.3f}, p={chi_p:.4f}')
    
    # ===== PREPARE FOR MODELING =====
    print('\n-> PREPARING DATA FOR MODELING')

    X_cls = df_cls.drop(columns=[target, target+'_Class', 'Temp_Cat'], errors='ignore')
    y_cls = df_cls[target+'_Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cls)
    
    X_train, X_test, y_train, y_test = mdl.split_data(X_scaled, y_cls, stratify=y_cls)
    print(f'Train: {len(X_train)}, Test: {len(X_test)} (80/20 split)')
    
    # regression data
    X_reg = df_model.drop(columns=[target])
    y_reg = df_model[target]
    X_reg_scaled = scaler.fit_transform(X_reg)
    X_tr_r, X_te_r, y_tr_r, y_te_r = mdl.split_data(X_reg_scaled, y_reg)
    
    # ===== FEATURE REDUCTION =====
    print('FEATURE REDUCTION')

    X_pca, pca = mdl.do_pca(X_scaled, n=2)
    print(f'PCA: {sum(pca.explained_variance_ratio_)*100:.1f}% variance explained')
    
    X_kpca, kpca = mdl.do_kernel_pca(X_scaled, n=2, kernel='rbf')
    print(f'Kernel PCA (RBF): 2 components extracted')
    
    X_lda, lda = mdl.do_lda(X_scaled, y_cls, n=2)
    print(f'LDA: 2 discriminant components')
    
    X_svd, svd = mdl.do_svd(X_scaled, n=2)
    print(f'SVD: {sum(svd.explained_variance_ratio_)*100:.1f}% variance explained')
    
    # ===== CLASSIFICATION MODELS =====
    print('\n-> CLASSIFICATION MODELS')

    clf_results = []
    classes = ['Low', 'Medium', 'High']
    
    classifiers = {
        'NaiveBayes': mdl.naive_bayes,
        'BayesianNetwork': mdl.bayesian_network,
        'DecisionTree': lambda X,y: mdl.decision_tree(X, y, depth=5),
        'KNN_euclidean': lambda X,y: mdl.knn(X, y, metric='euclidean'),
        'KNN_manhattan': lambda X,y: mdl.knn(X, y, metric='manhattan'),
        'KNN_minkowski': lambda X,y: mdl.knn(X, y, metric='minkowski'),
        'LDA': mdl.lda_classifier,
        'LogisticReg': mdl.logistic_reg,
        'NeuralNet': mdl.neural_net_clf
    }
    
    fit_status = {}
    
    for name, func in classifiers.items():
        print(f'\n  {name}:')
        m = func(X_train, y_train)
        pred = m.predict(X_test)
        
        # metrics
        met = evl.clf_metrics(y_test, pred)
        met['model'] = name
        clf_results.append(met)
        
        print(f'    Accuracy: {met["accuracy"]:.3f}')
        print(f'    Error Rate: {met["error_rate"]:.3f}')
        print(f'    Precision: {met["precision"]:.3f}')
        print(f'    Recall: {met["recall"]:.3f}')
        print(f'    F1: {met["f1"]:.3f}')
        
        # cross validation
        cv_mean, cv_std = mdl.cross_validate(m, X_scaled, y_cls)
        print(f'    CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}')
        
        # confusion matrix
        evl.plot_confusion(y_test, pred, m.classes_, f'{PLOTS_DIR}/cm_{name}.png')
        
        # roc curve
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(X_test)
            evl.plot_roc(y_test, proba, m.classes_, f'{PLOTS_DIR}/roc_{name}.png')
        
        # learning curve for overfitting check
        try:
            train_sizes, train_sc, test_sc = mdl.get_learning_curve(m, X_scaled, y_cls)
            status = evl.plot_learning_curve(train_sizes, train_sc, test_sc, name, f'{PLOTS_DIR}/lc_{name}.png')
            fit_status[name] = status
            print(f'    Fit Status: {status}')
        except:
            pass
    
    # ===== REGRESSION MODELS =====
    print('\n-> REGRESSION MODELS')

    reg_results = []
    
    regressors = {
        'LinearRegression': mdl.linear_reg,
        'NeuralNetRegressor': mdl.neural_net_reg
    }
    
    for name, func in regressors.items():
        print(f'\n  {name}:')
        m = func(X_tr_r, y_tr_r)
        pred = m.predict(X_te_r)
        
        met = evl.reg_metrics(y_te_r, pred)
        met['model'] = name
        reg_results.append(met)
        
        print(f'    MAE: {met["mae"]:.3f}')
        print(f'    RMSE: {met["rmse"]:.3f}')
        print(f'    R²: {met["r2"]:.3f}')
        print(f'    Willmott Index: {met["willmott"]:.3f}')
        print(f'    Nash-Sutcliffe: {met["nse"]:.3f}')
        print(f'    Legates-McCabe: {met["legates_mccabe"]:.3f}')
    
    # ===== SAVE RESULTS =====
    print('\n-> SAVING RESULTS')

    clf_df = pd.DataFrame(clf_results)
    clf_df.to_csv(f'{CSVS_DIR}/classification_results.csv', index=False)
    
    reg_df = pd.DataFrame(reg_results)
    reg_df.to_csv(f'{CSVS_DIR}/regression_results.csv', index=False)
    
    print(f'\nAll outputs saved to {OUT_DIR}/')
    print('\nClassification Summary:')
    print(clf_df[['model', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
    
    print('\nRegression Summary:')
    print(reg_df.to_string(index=False))
    
    print('\nPIPELINE COMPLETE!')

if __name__ == '__main__':
    main()
