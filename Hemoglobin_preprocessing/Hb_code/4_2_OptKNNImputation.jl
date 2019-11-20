using OptImpute
using OptimalTrees
using DataFrames

# Import data with missing values
X4 = readtable("data/X_not_imputed_4h.csv")
X24 = readtable("data/X_not_imputed_24h.csv")
Xf4 = readtable("data/X_filter_not_imputed_4h.csv")
Xf24 = readtable("data/X_filter_not_imputed_24h.csv")


# Imputation
X_opt_knn_4 = OptImpute.impute(X4, :opt_knn, knn_k = 10);
X_opt_knn_24 = OptImpute.impute(X24, :opt_knn, knn_k = 10);
X_opt_knn_f4 = OptImpute.impute(Xf4, :opt_knn, knn_k = 10);
X_opt_knn_f24 = OptImpute.impute(Xf24, :opt_knn, knn_k = 10);


writetable("data/X_knn_imputed_4h.csv", X_opt_knn_4)
writetable("data/X_knn_imputed_24h.csv", X_opt_knn_24)
writetable("data/X_filter_knn_imputed_4h.csv", X_opt_knn_f4)
writetable("data/X_filter_knn_imputed_24h.csv", X_opt_knn_f24)

