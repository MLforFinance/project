import pandas as pd
from sklearn.preprocessing import StandardScaler

from mrsq import mrsq
from factors_em import factors_em
from remove_outliers import remove_outliers
from prepare_missing import prepare_missing, compute_NA


csv_path = 'data/2026-02-MD.csv'
DEMEAN = 2
jj = 2
kmax = 8

# 1. LOAD DATA
df = pd.read_csv(csv_path)
series_names = df.columns[1:].values
tcode = df.iloc[0, 1:].values.astype(int)
rawdata = df.iloc[1:, 1:].values.astype(float)

print((compute_NA(df) == 0).all())

# 2. PROCESS
yt = prepare_missing(rawdata, tcode)
yt = yt[2:, :] # remove 2 first months because of second order diff

data, n_outliers = remove_outliers(yt)

# 3. ESTIMATE
ehat, Fhat, lamhat, ve2, x2 = factors_em(data, kmax, jj, DEMEAN)

R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq(Fhat, lamhat, ve2, series_names)

print(f"Total variance explained: {R2_T:.2%}")

transformed_df = pd.DataFrame(x2, columns=series_names)

print((compute_NA(transformed_df) == 0).all())

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(transformed_df), columns=series_names)

scaled_data.to_csv('data/2026-02-MD_processed.csv')