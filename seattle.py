from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd

listing = pd.read_csv("datasets/airbnb/seattle/listings.csv").head(1000)
print(listing.shape)
listing = listing.drop(['license'], axis=1)
print(listing.shape)
print(type(listing))
clean_listing = listing.dropna(how='any', thresh = 70)
print(type(clean_listing))
clean_listing = clean_listing.drop_duplicates()
print(clean_listing)


cleaner_listing = clean_listing.fillna(0)

cleaner_listing['host_response_rate'] = cleaner_listing['host_response_rate'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['host_acceptance_rate'] = cleaner_listing['host_acceptance_rate'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['price'] = cleaner_listing['price'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['weekly_price'] = cleaner_listing['weekly_price'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['monthly_price'] = cleaner_listing['monthly_price'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['security_deposit'] = cleaner_listing['security_deposit'].str.replace(r'\D', '').astype(np.float64)
cleaner_listing['cleaning_fee'] = cleaner_listing['cleaning_fee'].str.replace(r'\D', '').astype(np.float64)


cleaner_listing = cleaner_listing.fillna(0)

cleaner_listing = cleaner_listing[['neighbourhood', 'host_response_rate', 'host_acceptance_rate',
        'host_total_listings_count', 'host_identity_verified', 'is_location_exact', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
       'cleaning_fee', 'minimum_nights', 'maximum_nights', 'has_availability', 'number_of_reviews', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'calculated_host_listings_count', 'reviews_per_month', 'cancellation_policy',
       'instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification', 'review_scores_value']]
print(len(cleaner_listing.columns))

cleaner_listing['room_type'] = cleaner_listing['room_type'].replace('Entire home/apt', 1, regex=True).replace('Private room', 2, regex=True).replace('Shared room', 3, regex=True).replace('', 0, regex=True)
cleaner_listing['property_type'] = cleaner_listing['property_type'].replace(['Apartment', 'House', 'Cabin', 'Condominium', 'Camper/RV', 'Bungalow', 'Townhouse', 'Loft', 'Boat', 'Bed & Breakfast',
       'Other', ''], [1,2,3,4,5,6,7,8,9,10,11,0])
cleaner_listing['cancellation_policy'] = cleaner_listing['cancellation_policy'].replace(['moderate', 'strict', 'flexible', ''], [1,2,3,0])
cleaner_listing.bed_type = cleaner_listing.bed_type.replace(['Real Bed', 'Futon', 'Pull-out Sofa', 'Airbed', 'Couch', ''], [1,2,3,4,5,0])
cleaner_listing.neighbourhood = cleaner_listing.neighbourhood.replace(['Queen Anne', 0, 'Ballard', 'Phinney Ridge', 'Fremont', 'Lower Queen Anne', 'Westlake', 'Wallingford', 'Green Lake',
       'Minor', 'Madrona', 'Harrison/Denny-Blaine', 'Leschi', 'University District', 'Roosevelt', 'Madison Park', 'Capitol Hill', 'Atlantic', ''], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,0])

print(len(cleaner_listing.columns))

cleaner_listing.host_identity_verified = cleaner_listing.host_identity_verified.replace(['f','t'], [0,1])
cleaner_listing.is_location_exact = cleaner_listing.is_location_exact.replace(['f','t'], [0,1])
cleaner_listing.has_availability = cleaner_listing.has_availability.replace(['f','t'], [0,1])
cleaner_listing.instant_bookable = cleaner_listing.instant_bookable.replace(['f','t'], [0,1])
cleaner_listing.require_guest_profile_picture = cleaner_listing.require_guest_profile_picture.replace(['f','t'], [0,1])
cleaner_listing.require_guest_phone_verification = cleaner_listing.require_guest_phone_verification.replace(['f','t'], [0,1])

#############################################################################

yVar = cleaner_listing['review_scores_value']

cleaner_listing = cleaner_listing.drop(['review_scores_value'], axis = 1)

print(len(cleaner_listing.columns))


print("____________________________________________________________________________")
print(np.where(len(np.float64(cleaner_listing.values) >= np.finfo(np.float64).max)))
print("____________________________________________________________________________")

cleaner_listing = cleaner_listing.reset_index()

X_train, X_test, y_train, y_test = train_test_split(cleaner_listing, yVar, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

preds = clf.predict(X_test)
cleaner_listing['pred'] = preds

cleaner_listing.to_excel("Seattle(clean)1.xlsx", sheet_name='Sheet_name_1')
