import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score


print("""Prilikom testiranja zakljuceno je da Random Forest algoritam
zauzima previse resursa te ga je tesko primeniti na velikim setovima
podataka. Najveci problem su hardverska ogranicenja. Primer od 20000
podataka zauzme 32GB ram memorije.""")

print("""

Odabranu sa dva algoritma za smanjenje dimenzionalnosti:
    - Neighborhood Components Analysis (Metricki algoritam)
    - Principal Component Analysis (Smanjenje dimenzionalnosti)

Za model je koristen SGDClassifier jer on omogucava partial_fit,
odnosno da se dodatno dopunjuje model novim setom podataka.

""")

print("""
U nastavku cemo videti rezultate sa 3 razlicite dimenzije:
    - 2
    - 5
    - 10

Kao i nad razlicitim skupom podataka:
    -1%
    -2%
    -3%
    -5% (Ovo takodje poprilicno traje)
    -100% (Nece biti u live prikazu jer dugo traje)
""")

df = pd.read_csv('CovidData.csv').dropna()

df['DATE_DIED'] = df['DATE_DIED'].apply(lambda x: False if x == '9999-99-99' else True)

# Select features and target variable
features = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'AGE', 'PREGNANT',
            'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
            'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU']
target = 'DATE_DIED'

X = df[features]
y = df[target]

le = LabelEncoder()
for column in X.select_dtypes(include='object').columns:
    X[column] = le.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dimensions=[2,5,10]
size=[0.01,0.02,0.03,0.05,1]
size=[0.005,0.01,0.02,0.03]

for x in dimensions:
    for y in size:
        print("Dimenzija: "+str(x))
        print("Velicina skupa: "+str(100*y)+"%")
        # Initialize the models
        scaler = StandardScaler()
        pca = PCA(n_components=x)
        nca = NeighborhoodComponentsAnalysis(n_components=x, random_state=42)
        sgd_model = SGDClassifier()
        sgd_model_nca = SGDClassifier()

        chunk_size = 1000  
        for i in range(0, int(len(X_train)*y), chunk_size): #len(X_train)
            X_chunk = X_train.iloc[i:i+chunk_size]
            y_chunk = y_train.iloc[i:i+chunk_size]

            X_chunk_scaled = scaler.fit_transform(X_chunk)

            # Apply PCA
            X_chunk_pca = pca.fit_transform(X_chunk_scaled)

            # Apply NCA 
            X_chunk_nca = nca.fit_transform(X_chunk_scaled, y_chunk)

            sgd_model.partial_fit(X_chunk_pca, y_chunk, classes=[0, 1])

            sgd_model_nca.partial_fit(X_chunk_nca, y_chunk, classes=[0, 1])


        # Make predictions on the test data
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        sgd_predictions = sgd_model.predict(X_test_pca)

        X_test_nca = pca.transform(X_test_scaled)
        sgd_predictions_nca = sgd_model_nca.predict(X_test_nca)

        # Evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, sgd_predictions)
        print(f'\tTacnost predikcije PCA: {accuracy}')

        accuracy = accuracy_score(y_test, sgd_predictions_nca)
        print(f'\tTacnost predikcije NCA: {accuracy}')



print("""

Zakljucak:
    Povecanjem dimenzionalnosti povecavamo tacnost predikcije, medjutim usporavamo performans.
    Povecanjem skupa podataka takodje povecavamo tacnost, medjutim takodje usporavamo preformans.
    Potrebno je pronaci neku sredinu izmedju dimenzionalnosti i velicine skupa za trening.
    Ukoliko previse smanjimo dimenzionalnost mozemo odvesti algoritam u pogresnu stranu i drasticno
    smanjiti njegovu tacnost.""")