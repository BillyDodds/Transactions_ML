
import unittest 
import process_data as model
import scraper, load_data
from nltk.corpus import stopwords # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from my_exceptions import InvalidDataFrameFormat



class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.blacklist = stopwords.words('english')
        self.blacklist += ['card', 'aus', 'au', 'ns', 'nsw', 'xx', 'pty', 'ltd', 'nswau']
        self.test_data = pd.read_csv("../files/test_data.csv", index_col=0)
        self.easy_corpus = {
            'test': [
                'mcdonalds', 'chargrill', 'charlies', 'blooms', 'quattro', 'bws', 'berry'
            ]
        }
        self.test_corpus = {
            'food': [
                'mcd', 'nsw', 'cha', 'mcdo', 'nswa', 'char', 'mcdon', 'nswau', 'mcdona',
                'mcdonal', 'mcdonald', 'mcdonalds', 'coo', 'lan', 'cov', 'qua', 'piz', 'ber', 'alb',
                'par', 'coog', 'lane', 'cove', 'quat', 'pizz', 'berr', 'albi', 'park', 'cooge', 'charg',
                'charl', 'quatt', 'pizze', 'berry', 'albio', 'coogee', 'chargr', 'charli', 'quattr',
                'pizzer', 'albion', 'chargri', 'charlie', 'quattro', 'pizzeri', 'chargril', 'charlies',
                'pizzeria', 'chargrill'
            ], 
            'life/wellbeing': [
                'blo', 'che', 'now', 'bloo', 'chem', 'nowr','bloom', 'chemi', 'nowra',
                'blooms', 'chemis', 'chemist'
            ], 
            'beers': [
                'bws', 'liq', 'new', 'liqu', 'newt', 'liquo', 'newto', 'liquor', 'newtow','newtown'
            ], 
            'shopping': [
                'ber', 'berr', 'berry', 'iga', 'plu', 'liq', 'plus', 'liqu', 'liquo'
            ]
        }

    def test_clean(self):
        result = model.clean("nsw we're a b c xx no stran3gers to aus card love pty ltd", self.blacklist)
        self.assertEqual(result, "stran gers love")
    
    def test_clean_chop(self):
        result = model.clean_chop("you a b c xx know the 1350 rules aus card and so pty do i ltd", self.blacklist)
        self.assertEqual(result, "know kno rules rule rul")

    def test_get_corpora(self):
        # Test whether try/except block is triggered when relevant columns are dropped
        with self.assertRaises(InvalidDataFrameFormat): 
            model.get_corpora(self.test_data.drop('category', axis=1))
        
        with self.assertRaises(InvalidDataFrameFormat):
            model.get_corpora(self.test_data.drop('desc_corpus', axis=1))
        

        # Tests whether all goes to plan when irrelevant columns are dropped and when only relevant columns are present.
        only_relevant_cols = model.get_corpora(self.test_data[['category', 'desc_corpus']])
        missing_irrelevant_cols = model.get_corpora(self.test_data.drop('description', axis=1))
        intended_corpora = model.get_corpora(self.test_data)
        self.assertEqual(missing_irrelevant_cols, intended_corpora)
        self.assertEqual(only_relevant_cols, intended_corpora)

        # Test whether all categories are present in the keys of the corpora.
        self.assertEqual(list(intended_corpora.keys()), list(self.test_data.category.unique()))
    
    def test_distance(self):
        # return 1 when two strings are completely different
        different = model.distance("rick", "astley")
        self.assertEqual(different, 1.0)

        # return 0 when two strings are identical
        same = model.distance("never gonna give", "never gonna give")
        self.assertEqual(same, 0.0)

        # test function against manual calculations (1 - intersection/union)
        dist = model.distance("desert you", "hurt you")
        pred_dist = 1 - (6/(6+4)) # = 0.4
        self.assertEqual(dist, pred_dist)
        desert_set = set("desert you")
        hurt_set = set("hurt you")
        intersect = len(desert_set.intersection(hurt_set))
        union = len(desert_set.union(hurt_set))
        calc_pred_dist = 1 - intersect/union
        self.assertEqual(dist, calc_pred_dist)
        
    def test_desc_dist(self):
        # if all elements of the features are present in corpus, then desc_dist should be zero
        all_present = model.desc_dist(["We've", "known", "each", "other", "for", "so", "long"], "known for other")
        self.assertEqual(all_present, 0)

        # if two of three elements of the features are present in corpus, and one is completely different from everything, should return 1/3 (average of [0,0,1])
        one_outlier = model.desc_dist(["Inside", "we", "both", "know", "what's", "been", "going", "on"], "going zzz both")
        self.assertEqual(one_outlier, 1/3)

        # check handling of empty corpora
        empty_corpus = model.desc_dist([], 'Never gonna tell a lie and hurt you')
        self.assertEqual(empty_corpus, 100)

        # check handling of empty features
        empty_features = model.desc_dist(["Inside", "we", "both", "know", "what's", "been", "going", "on"], '')
        self.assertEqual(empty_features, 1)
    
    def test_NLP_distances(self):
        data_with_distances = model.NLP_distances(self.test_data, self.test_corpus)

        # ensure all categories in corpus appear as a column in the resultant dataframe
        expected_columns = [cat + "_desc_dist" for cat in self.test_corpus.keys()]
        self.assertTrue(set(expected_columns).issubset(set(data_with_distances.columns)))

        # check handling of missing columns
        with self.assertRaises(InvalidDataFrameFormat):
            model.NLP_distances(self.test_data.drop("description", axis=1), self.test_corpus)
        
        with self.assertRaises(InvalidDataFrameFormat):
            model.NLP_distances(self.test_data.drop(["description", "date", "desc_features", "desc_corpus"], axis=1),self.test_corpus)

        # check desc_dist columns give the correct values for their intended rows
        easy_dists = model.NLP_distances(self.test_data, self.easy_corpus)
        exp_values = np.array([3/4, 2/4, 2/3, 1/3, 2/3, 3/5, 3/4])
        act_values = np.array(easy_dists.test_desc_dist)
        self.assertTrue( np.all(exp_values == act_values) )
    
    def test_scale(self):
        easy_dists = model.NLP_distances(self.test_data.drop("category", axis=1), self.easy_corpus)
        easy_values = np.array(easy_dists.test_desc_dist)

        # expected scaling
        µ = np.mean(easy_values)
        sd = np.std(easy_values)
        exp_scale = (easy_values - µ)/sd

        # actual scaling
        scaler = StandardScaler()
        scaler.fit(easy_dists)
        easy_scale = model.scale(easy_dists, scaler)
        act_scale = easy_scale.test_desc_dist

        # assert equality
        self.assertTrue( np.all(act_scale == exp_scale) )
    
    def test_run_fold(self):
        easy_data = self.test_data
        ml_model = DecisionTreeClassifier()

        # Checks on a deterministic sample whether the result is working as intended
        test_split_1 = (np.array(range(4)), np.array([4, 5, 6]))
        result = model.run_fold(easy_data, test_split_1, ml_model, verbose=False)
        self.assertEqual(result, 1/3)

        test_split_2 = (np.array(range(5)), np.array([5, 6]))
        result = model.run_fold(easy_data, test_split_2, ml_model, verbose=False)
        self.assertEqual(result, 0.5)

        # Checks whether model has 100% in-sample accuracy (it should)
        test_split_3 = (np.array(range(7)), np.array(range(7)))
        result = model.run_fold(easy_data, test_split_3, ml_model, verbose=False)
        self.assertEqual(result, 1.0)

# class TestScraper(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         pass
    
#     def test_google(self):
#         pass


if __name__ == '__main__':
    unittest.main()