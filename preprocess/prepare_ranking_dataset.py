import itertools
import os
import json
import pandas as pd
import logging
import random

from conf import configuration

difficulty = "easy" # "easy" or "hard"
# train_dataset = ['movies', 'books', 'music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'rocks']
# train_dataset = ['music', 'odours', 'rocks']
# train_dataset = ['physical', 'music']
# test_dataset = ['tastes']
# test_dataset = ['rocks']

# # For tastes domain
# train_dataset = ['movies', 'books', 'music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'rocks']
# train_dataset = ['music', 'odours', 'rocks']
# test_dataset = ['tastes']
# num_test_examples_per_dataset = 2040

# # For rocks domain
# train_dataset = ['movies', 'books', 'music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'tastes']
# # train_dataset = ['music', 'odours', 'tastes']
# test_dataset = ['rocks']
# num_test_examples_per_dataset = 2380

# # For odours domain
# train_dataset = ['movies', 'books', 'music', 'wikidata1', 'wikidata2', 'physical', 'tastes', 'rocks']
# # train_dataset = ['music', 'tastes', 'rocks']
# test_dataset = ['odours']
# num_test_examples_per_dataset = 1360


# # For music domain
# train_dataset = ['movies', 'books', 'odours', 'wikidata1', 'wikidata2', 'physical', 'tastes', 'rocks']
# # train_dataset = ['odours', 'tastes', 'rocks']
# test_dataset = ['music']
# num_test_examples_per_dataset = 3060



# # For movies domain
# # train_dataset = ['music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'tastes', 'rocks']
# train_dataset = ['odours', 'music', 'tastes', 'rocks']
# test_dataset = ['movies']
# num_test_examples_per_dataset = 500


# # For books domain
# train_dataset = ['music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'tastes', 'rocks']
# # train_dataset = ['odours', 'music', 'tastes', 'rocks']
# test_dataset = ['books']
# num_test_examples_per_dataset = 500


# For wikidata1 domain
# train_dataset = ['music', 'odours', 'movies', 'books', 'physical', 'tastes', 'rocks']
train_dataset = ['odours', 'music', 'tastes', 'rocks']
test_dataset = ['wikidata1']
num_test_examples_per_dataset = 500


# # For wikidata2 domain
# train_dataset = ['music', 'odours', 'movies', 'books', 'physical', 'tastes', 'rocks']
# # train_dataset = ['odours', 'music', 'tastes', 'rocks']
# test_dataset = ['wikidata2']
# num_test_examples_per_dataset = 500



# # For physical domain
# train_dataset = ['music', 'odours', 'movies', 'books', 'wikidata1', 'wikidata2', 'tastes', 'rocks']
# # train_dataset = ['odours', 'music', 'tastes', 'rocks']
# test_dataset = ['physical']
# num_test_examples_per_dataset = 500






topPopular = 100
num_train_examples_per_dataset = 500

# num_test_examples_per_dataset = 2040
# num_test_examples_per_dataset = 280

train_dataset_str = '+'.join(train_dataset)
test_dataset_str = '+'.join(test_dataset)

# train_output_file = os.path.join(configuration.data_folder, f'train_{difficulty}_ranking_dataset_{train_dataset_str}_{num_train_examples_per_dataset}.jsonl')
# test_output_file = os.path.join(configuration.data_folder, f'test_{difficulty}_ranking_dataset_{test_dataset_str}_{num_test_examples_per_dataset}.jsonl')


train_output_file = os.path.join(configuration.data_folder, f'cross_domain_{difficulty}_train_{train_dataset_str}_{num_train_examples_per_dataset}.jsonl')
test_output_file = os.path.join(configuration.data_folder, f'cross_domain_{difficulty}_test_{test_dataset_str}_{num_test_examples_per_dataset}.jsonl')



random.seed(42)

def generate_question(question, element1, element2):
    return question.format(element1=element1, element2=element2)

def get_data_files_declarative_bias(train_dataset):
    data_files_declarative_bias = []
    if 'wikidata1' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'WikiDataMainProperties/matched_outputRiver_WikiPageRank.txt',
                "entity_type": "river",
                "property_header": "length",
                "entity_header": "riverLabel",
                "property": "long river",
                "high property": "very long river",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two rivers: Is {element1} longer than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/matched_query_cityRankedPopulation_WikiRank.txt',
                "entity_type": "city",
                "property_header": "maxPopulation",
                "entity_header": "cityLabel",
                "property": "populous city",
                "high property": "very populous city",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two cities: Is {element1} more populous than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/matched_query_PersonBorn_London_WikiRank.txt',
                "entity_type": "person",
                "property_header": "dob",
                "entity_header": "personLabel",
                "property": "recently born personality",
                "high property": "very recently born personality",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two people: Is {element1} more recently born than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/matched_query_Rank_BuildingHeight_WikiRank.txt',
                "entity_type": "building",
                "property_header": "maxHeight",
                "entity_header": "itemLabel",
                "property": "tall building",
                "high property": "very tall building",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two buildings: Is {element1} taller than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/matched_query_Rank_Island_Area_WikiRank.txt',
                "entity_type": "island",
                "property_header": "No_islandArea",
                "entity_header": "islandLabel",
                "property": "large island",
                "high property": "very large island",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two islands: Is {element1} larger than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/matched_RankMusueumsLattitude_Italy_WikiRank.txt',
                "entity_type": "museum",
                "property_header": "Rank_lat",
                "entity_header": "museumLabel",
                "property": "northern museum in Italy",
                "high property": "most northern museum in Italy",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two museums: Is {element1} more northern than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/unique_matched_InceptionCompanyWikiPageRank.txt',
                "entity_type": "company",
                "property_header": "minInception",
                "entity_header": "companyLabel",
                "property": "recently founded company",
                "high property": "very recently founded company",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two companies: Is {element1} more recently founded than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/unique_matched_MountainHeightWikiPageRank.txt',
                "entity_type": "mountain",
                "property_header": "elevation",
                "entity_header": "mountainLabel",
                "property": "tall mountain",
                "high property": "very tall mountain",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two mountains: Is {element1} taller than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/unique_matched_Person_SOcialMedia_WikiPageRank.txt',
                "entity_type": "person",
                "property_header": "maxSocialMediaFollower",
                "entity_header": "personLabel",
                "property": "personality with high social media followers",
                "high property": "personality with very high social media followers",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two personalities: Is {element1} more popular than {element2}?"
            },
            {
                "file_path": 'WikiDataMainProperties/unique_matched_speciesMass_WikiPageRank.txt',
                "entity_type": "species",
                "property_header": "maxMass",
                "entity_header": "speciesLabel",
                "property": "heavy species",
                "high property": "very heavy species",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two species: Is {element1} heavier than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'wikidata2' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'WikiDataSubtleProperties/matched_query_chemcialELements_DIscovery_WikiPageRank.txt',
                "entity_type": "element",
                "property_header": "minDiscovery",
                "entity_header": "elementLabel",
                "property": "long-known element",
                "high property": "much long-known element",
                "ascending": True,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two elements: Is {element1} discovered earlier than {element2}?"
            },
            {
                "file_path": 'WikiDataSubtleProperties/matched_queryfood_WaterFootPrint_WikiRank.txt',
                "entity_type": "food",
                "property_header": "WaterFootPrint",
                "entity_header": "foodGrpLabel",
                "property": "high water footprint food",
                "high property": "very high water footprint food",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two food items: Does {element1} have a larger water footprint than {element2}?"
            },
            {
                "file_path": 'WikiDataSubtleProperties/matched_query_rankCountries_Population.txt',
                "entity_type": "country",
                "property_header": "maxPopulation",
                "entity_header": "countryLabel",
                "property": "populous country",
                "high property": "very populous country",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two countries: Is {element1} more populous than {element2}?"
            },
            {
                "file_path": 'WikiDataSubtleProperties/matched_query_rankElements_AtomicNo_WikiPageRank.txt',
                "entity_type": "chemical element",
                "property_header": "atomicNo",
                "entity_header": "elementLabel",
                "property": "high atomic number element",
                "high property": "very high atomic number element",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two chemical elements: Does {element1} have a higher atomic number than {element2}?"
            },
            {
                "file_path": 'WikiDataSubtleProperties/matched_query_schoville_WikiRank.txt',
                "entity_type": "food",
                "property_header": "Rank_scovilleGrade",
                "entity_header": "foodName",
                "property": "spicy food",
                "high property": "very spicy food",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": False,
                "question": "This question is about two food items: Is {element1} spicier than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'movies' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Movie/NEWFunny_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "funny movie",
                "high property": "very funny movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} funnier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWAbsurd_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "absurd movie",
                "high property": "very absurd movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more absurd than {element2}?"
            },
            {
                "file_path": 'Movie/NEWArtistic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "artistic movie",
                "high property": "very artistic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more artistic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWBeautiful_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "beautiful movie",
                "high property": "very beautiful movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more beautiful than {element2}?"
            },
            {
                "file_path": 'Movie/NEWBleak_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "bleak movie",
                "high property": "very bleak movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} bleaker than {element2}?"
            },
            {
                "file_path": 'Movie/NEWBloody_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "bloody movie",
                "high property": "very bloody movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} bloodier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWBoring_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "boring movie",
                "high property": "very boring movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more boring than {element2}?"
            },
            {
                "file_path": 'Movie/NEWclaustrophobic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "claustrophobic movie",
                "high property": "very claustrophobic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more claustrophobic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWClever_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "clever movie",
                "high property": "very clever movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more clever than {element2}?"
            },
            {
                "file_path": 'Movie/NEWColourful_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "colourful movie",
                "high property": "very colourful movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more colourful than {element2}?"
            },
            {
                "file_path": 'Movie/NEWComplex_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "complex movie",
                "high property": "very complex movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more complex than {element2}?"
            },
            {
                "file_path": 'Movie/NEWConfrontational_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "confrontational movie",
                "high property": "very confrontational movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more confrontational than {element2}?"
            },
            {
                "file_path": 'Movie/NEWDark_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "dark movie",
                "high property": "very dark movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} darker than {element2}?"
            },
            {
                "file_path": 'Movie/NEWDramatic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "dramatic movie",
                "high property": "very dramatic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more dramatic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWEducational_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "educational movie",
                "high property": "very educational movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more educational than {element2}?"
            },
            {
                "file_path": 'Movie/NEWEmotional_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "emotional movie",
                "high property": "very emotional movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more emotional than {element2}?"
            },
            {
                "file_path": 'Movie/NEWEnigmatic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "enigmatic movie",
                "high property": "very enigmatic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more enigmatic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWFrightening_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "frightening movie",
                "high property": "very frightening movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more frightening than {element2}?"
            },
            {
                "file_path": 'Movie/NEWControversial_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "controversial movie",
                "high property": "very controversial movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more controversial than {element2}?"
            },
            {
                "file_path": 'Movie/NEWGory_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "gory movie",
                "high property": "very gory movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} gorier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWGrim_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "grim movie",
                "high property": "very grim movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} grimmer than {element2}?"
            },
            {
                "file_path": 'Movie/NEWGritty_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "gritty movie",
                "high property": "very gritty movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} grittier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWGruesome_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "gruesome movie",
                "high property": "very gruesome movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more gruesome than {element2}?"
            },
            {
                "file_path": 'Movie/NEWInspirational_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "inspirational movie",
                "high property": "very inspirational movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more inspirational than {element2}?"
            },
            {
                "file_path": 'Movie/NEWIntellectual_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "intellectual movie",
                "high property": "very intellectual movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more intellectual than {element2}?"
            },
            {
                "file_path": 'Movie/NEWIntelligent_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "intelligent movie",
                "high property": "very intelligent movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more intelligent than {element2}?"
            },
            {
                "file_path": 'Movie/NEWIntense_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "intense movie",
                "high property": "very intense movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more intense than {element2}?"
            },
            {
                "file_path": 'Movie/NEWMelancholic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "melancholic movie",
                "high property": "very melancholic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more melancholic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWMoody_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "moody movie",
                "high property": "very moody movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} moodier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWPredictable_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "predictable movie",
                "high property": "very predictable movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more predictable than {element2}?"
            },
            {
                "file_path": 'Movie/NEWPretentious_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "pretentious movie",
                "high property": "very pretentious movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more pretentious than {element2}?"
            },
            {
                "file_path": 'Movie/NEWQuirky_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "quirky movie",
                "high property": "very quirky movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} quirkier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWRealistic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "realistic movie",
                "high property": "very realistic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more realistic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWRomantic_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "romantic movie",
                "high property": "very romantic movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more romantic than {element2}?"
            },
            {
                "file_path": 'Movie/NEWSad_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "sad movie",
                "high property": "very sad movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} sadder than {element2}?"
            },
            {
                "file_path": 'Movie/NEWSatirical_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "satirical movie",
                "high property": "very satirical movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more satirical than {element2}?"
            },
            {
                "file_path": 'Movie/NEWScary_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "scary movie",
                "high property": "very scary movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} scarier than {element2}?"
            },
            {
                "file_path": 'Movie/NEWSentimental_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "sentimental movie",
                "high property": "very sentimental movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more sentimental than {element2}?"
            },
            {
                "file_path": 'Movie/NEWStunning_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "stunning movie",
                "high property": "very stunning movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more stunning than {element2}?"
            },
            {
                "file_path": 'Movie/NEWSurreal_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "surreal movie",
                "high property": "very surreal movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more surreal than {element2}?"
            },
            {
                "file_path": 'Movie/NEWSuspenseful_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "suspenseful movie",
                "high property": "very suspenseful movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more suspenseful than {element2}?"
            },
            {
                "file_path": 'Movie/NEWTense_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "tense movie",
                "high property": "very tense movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more tense than {element2}?"
            },
            {
                "file_path": 'Movie/NEWViolent_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "violent movie",
                "high property": "very violent movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} more violent than {element2}?"
            },
            {
                "file_path": 'Movie/NEWWitty_movie_titles1.txt',
                "entity_type": "movie",
                "property_header": "score",
                "entity_header": "title",
                "property": "witty movie",
                "high property": "very witty movie",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two movies: Is {element1} wittier than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'books' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Books/NEW_witty_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "witty book",
                "high property": "very witty book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} wittier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_weird_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "weird book",
                "high property": "very weird book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} weirder than {element2}?"
            },
            {
                "file_path": 'Books/NEW_unique_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "unique book",
                "high property": "very unique book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more unique than {element2}?"
            },
            {
                "file_path": 'Books/NEW_surreal_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "surreal book",
                "high property": "very surreal book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more surreal than {element2}?"
            },
            {
                "file_path": 'Books/NEW_strange_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "strange book",
                "high property": "very strange book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} stranger than {element2}?"
            },
            {
                "file_path": 'Books/NEW_silly_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "silly book",
                "high property": "very silly book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} sillier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_short_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "short book",
                "high property": "very short book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} shorter than {element2}?"
            },
            # Very less data for the below property
            {
                "file_path": 'Books/NEW_satirical_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "satirical book",
                "high property": "very satirical book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more satirical than {element2}?"
            },
            {
                "file_path": 'Books/NEW_sad_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "sad book",
                "high property": "very sad book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} sadder than {element2}?"
            },
            {
                "file_path": 'Books/NEW_romantic_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "romantic book",
                "high property": "very romantic book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more romantic than {element2}?"
            },
            {
                "file_path": 'Books/NEW_realistic_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "realistic book",
                "high property": "very realistic book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more realistic than {element2}?"
            },
            # Very less data for the below property
            {
                "file_path": 'Books/NEW_quirky_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "quirky book",
                "high property": "very quirky book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} quirkier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_predictable_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "predictable book",
                "high property": "very predictable book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more predictable than {element2}?"
            },
            {
                "file_path": 'Books/NEW_political_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "political book",
                "high property": "very political book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more political than {element2}?"
            },
            {
                "file_path": 'Books/NEW_absurd_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "absurd book",
                "high property": "very absurd book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more absurd than {element2}?"
            },
            {
                "file_path": 'Books/NEW_beautiful_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "beautiful book",
                "high property": "very beautiful book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more beautiful than {element2}?"
            },
            {
                "file_path": 'Books/NEW_bizarre_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "bizarre book",
                "high property": "very bizarre book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more bizarre than {element2}?"
            },
            {
                "file_path": 'Books/NEW_controversial_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "controversial book",
                "high property": "very controversial book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more controversial than {element2}?"
            },
            {
                "file_path": 'Books/NEW_cool_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "cool book",
                "high property": "very cool book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} cooler than {element2}?"
            },
            {
                "file_path": 'Books/NEW_crazy_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "crazy book",
                "high property": "very crazy book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} crazier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_dark_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "dark book",
                "high property": "very dark book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} darker than {element2}?"
            },
            {
                "file_path": 'Books/NEW_educational_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "educational book",
                "high property": "very educational book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more educational than {element2}?"
            },
            {
                "file_path": 'Books/NEW_funny_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "funny book",
                "high property": "very funny book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} funnier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_futuristic_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "futuristic book",
                "high property": "very futuristic book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more futuristic than {element2}?"
            },
            {
                "file_path": 'Books/NEW_gritty_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "gritty book",
                "high property": "very gritty book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} grittier than {element2}?"
            },
            {
                "file_path": 'Books/NEW_hilarious_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "hilarious book",
                "high property": "very hilarious book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more hilarious than {element2}?"
            },
            {
                "file_path": 'Books/NEW_inspirational_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "inspirational book",
                "high property": "very inspirational book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more inspirational than {element2}?"
            },
            {
                "file_path": 'Books/NEW_intellectual_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "intellectual book",
                "high property": "very intellectual book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more intellectual than {element2}?"
            },
            {
                "file_path": 'Books/NEW_intense_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "intense book",
                "high property": "very intense book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more intense than {element2}?"
            },
            {
                "file_path": 'Books/NEW_interesting_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "interesting book",
                "high property": "very interesting book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more interesting than {element2}?"
            },
            {
                "file_path": 'Books/NEW_literary_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "literary book",
                "high property": "very literary book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more literary than {element2}?"
            },
            {
                "file_path": 'Books/NEW_philosophical_Book_titles1.txt',
                "entity_type": "book",
                "property_header": "score",
                "entity_header": "title",
                "property": "philosophical book",
                "high property": "very philosophical book",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two books: Is {element1} more philosophical than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'tastes' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Sweet_Mean",
                "entity_header": "foodLabel",
                "property": "sweet taste food item",
                "high property": "very sweet taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally sweeter in taste than {element2}?"
            },
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Salty_Mean",
                "entity_header": "foodLabel",
                "property": "salty taste food item",
                "high property": "very salty taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally saltier in taste than {element2}?"
            },
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Sour_Mean",
                "entity_header": "foodLabel",
                "property": "sour taste food item",
                "high property": "very sour taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally sourer in taste than {element2}?"
            },
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Bitter_Mean",
                "entity_header": "foodLabel",
                "property": "bitter taste food item",
                "high property": "very bitter taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally more bitter in taste than {element2}?"
            },
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Umami_Mean",
                "entity_header": "foodLabel",
                "property": "umami taste food item",
                "high property": "very umami taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally more umami in taste than {element2}?"
            },
            {
                "file_path": 'Food_taste/food_Taste.txt',
                "entity_type": "food item",
                "property_header": "Fat_Mean",
                "entity_header": "foodLabel",
                "property": "fatty taste food item",
                "high property": "very fatty taste food item",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two food items: Is {element1} generally fattier in taste than {element2}?"
            },
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'rocks' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "lightness",
                "entity_header": "rockLabel",
                "property": "light colored type of rock",
                "high property": "very light colored type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} lighter in color than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "grainSize",
                "entity_header": "rockLabel",
                "property": "coarse-grained type of rock",
                "high property": "very coarse-grained type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} coarser in grain size than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "roughness",
                "entity_header": "rockLabel",
                "property": "rough textured type of rock",
                "high property": "very rough textured type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} rougher in texture than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "shine",
                "entity_header": "rockLabel",
                "property": "shiny type of rock",
                "high property": "very shiny type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} more shiny than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "organization",
                "entity_header": "rockLabel",
                "property": "uniform-textured type of rock",
                "high property": "very uniform-textured type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} more uniform in texture than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "variability",
                "entity_header": "rockLabel",
                "property": "variable-colored type of rock",
                "high property": "very variable-colored type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} more variable in color than {element2}?"
            },
            {
                "file_path": 'Rock/rock_data.txt',
                "entity_type": "rock",
                "property_header": "density",
                "entity_header": "rockLabel",
                "property": "dense type of rock",
                "high property": "very dense type of rock",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two types of rocks: Is {element1} denser than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'odours' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Odour/odour_dataset.txt',
                "entity_type": "thing",
                "property_header": "pleasantness",
                "entity_header": "items",
                "property": "pleasant odour",
                "high property": "very pleasant odour",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two things: Is {element1} more pleasant in odour than {element2}?"
            },
            {
                "file_path": 'Odour/odour_dataset.txt',
                "entity_type": "thing",
                "property_header": "intensity",
                "entity_header": "items",
                "property": "intense odour",
                "high property": "very intense odour",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two things: Is {element1} more intense in odour than {element2}?"
            },
            {
                "file_path": 'Odour/odour_dataset.txt',
                "entity_type": "thing",
                "property_header": "irritability",
                "entity_header": "items",
                "property": "irritating odour",
                "high property": "very irritating odour",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two things: Is {element1} more irritating in odour than {element2}?"
            },
            {
                "file_path": 'Odour/odour_dataset.txt',
                "entity_type": "thing",
                "property_header": "familiarity",
                "entity_header": "items",
                "property": "familiar odour",
                "high property": "very familiar odour",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two things: Is {element1} more familiar in odour than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'music' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "wond",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of wonder",
                "high property": "music that evokes the feeling of wonder",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more wonderful than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "tran",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of tranquility",
                "high property": "music that evokes the feeling of tranquility",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more tranquil than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "tend",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of tenderness",
                "high property": "music that evokes the feeling of tenderness",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more tender than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "nost",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of nostalgia",
                "high property": "music that evokes the feeling of nostalgia",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more nostalgic than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "peac",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of peace",
                "high property": "music that evokes the feeling of peace",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more peaceful than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "joya",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of joy",
                "high property": "music that evokes the feeling of joy",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more joyful than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "ener",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of energy",
                "high property": "music that evokes the feeling of energy",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more energetic than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "sadn",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of sadness",
                "high property": "music that evokes the feeling of sadness",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more sad than {element2}?"
            },
            {
                "file_path": 'Music/music_perception.txt',
                "entity_type": "music",
                "property_header": "tens",
                "entity_header": "artist - title",
                "property": "music that evokes the feeling of tension",
                "high property": "music that evokes the feeling of tension",
                "ascending": False,
                "ranking_header": "score",
                "selectAll": True,
                "question": "This question is about two music pieces: Is {element1} more tense than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    if 'physical' in train_dataset:
        declarative_bias = [
            {
                "file_path": 'PhysicalProperties/Mass/49_objectMass.txt',
                "entity_type": "entity",
                "property_header": "mass",
                "entity_header": "item",
                "property": "heavy entity",
                "high property": "very heavy entity",
                "ascending": False,
                "ranking_header": "QRank",
                "selectAll": True,
                "question": "This question is about two entities: Is {element1} heavier than {element2}?"
            }
        ]
        data_files_declarative_bias.extend(declarative_bias)
    return data_files_declarative_bias

def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

def create_physical_properties_examples(num_of_examples_for_adjacent_pairs, file_path, property, high_property, low_property, question):
    num_of_total_examples = num_of_examples_for_adjacent_pairs
    examples = []
    data = []
    with open(os.path.join(configuration.ranking_dataset_path, file_path), 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data.append(obj)
    random.shuffle(data)
    for item in data[0:num_of_total_examples]:
        entity1 = item["obj_a"]
        entity2 = item["obj_b"]
        example = {
                "entity_type": "entity",
                "entity1": entity1,
                "entity2": entity2,
                "relation": "large",
                "answer": "Yes" if item["label"] == 1 else "No",
                "property": property,
                "high_property": high_property,
                "low_property": low_property,
                "question": generate_question(question, entity1, entity2)
            }
        examples.append(example)
    return examples

def select_physical_properties_dataset(num_of_examples_for_adjacent_pairs):
    file_path = ['PhysicalProperties/500_pairwiseHeight.txt', 'PhysicalProperties/500_pairwiseSize.txt']
    property = ["tall entity", "large size entity"]
    high_property = ["very tall entity", "very large size entity"]
    question = ["This question is about two entities: Is {element1} taller than {element2}?",
                "This question is about two entities: Is {element1} larger than {element2}?"]
    examples = []
    n = len(file_path)
    for i in range(n):
        examples.extend(create_physical_properties_examples(num_of_examples_for_adjacent_pairs, file_path[i], property[i], high_property[i], low_property=high_property[i], question=question[i]))
    return examples

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def compare_entities(entity1, entity2, ascending):
    entity1 = float(entity1)
    entity2 = float(entity2)
    if ascending:
        return entity1 < entity2
    else:
        return entity1 > entity2

def create_adjacency_examples(next_ranked_prop_entities, num_of_examples, entity_type=None, relation=None, property=None, entity_header=None, property_header=None, ascending=None, high_property=None, low_property=None, question=None):
    examples = []
    n = len(next_ranked_prop_entities)

    # For hard difficulty, we restrict the pairs to be adjacent
    if difficulty == "hard":
        indices = list(range(n))
        adjacent_pairs = [(i, j) for i in indices for j in indices if i == j + 1 or i == j - 1]
        next_to_next_pairs = [(i, j) for i in indices for j in indices if i == j + 2 or i == j - 2]
        try:
            sampled_pairs = random.sample(adjacent_pairs, num_of_examples)
        except ValueError:
            print(f"Hard difficulty: {num_of_examples} to sample from adjacent list of len {len(adjacent_pairs)} for entity type: {entity_type} and property: {property}")
            logging.info(f"Hard difficulty: {num_of_examples} to sample from adjacent list of len {len(adjacent_pairs)} for entity type: {entity_type} and property: {property}")
            sampled_pairs = adjacent_pairs
        m = len(sampled_pairs)
        if m < num_of_examples:
            try:
                sampled_pairs.extend(random.sample(next_to_next_pairs, num_of_examples - m))
            except ValueError:
                print(f"Hard difficulty: {num_of_examples - m} to sample from next_to_next list of len {len(next_to_next_pairs)} for entity type: {entity_type} and property: {property}")
                logging.info(f"Hard difficulty: {num_of_examples - m} to sample from next_to_next list of len {len(next_to_next_pairs)} for entity type: {entity_type} and property: {property}")
                sampled_pairs.extend(next_to_next_pairs)

    # if difficulty == "hard":
    #     adjacent_pairs = []
    #     for i in range(n - 1):
    #         if random.choice([True, False]):
    #             adjacent_pairs.append((i, i + 1))
    #         else:
    #             adjacent_pairs.append((i + 1, i))
    #     next_to_next_pairs = []
    #     for i in range(n - 2):
    #         if random.choice([True, False]):
    #             next_to_next_pairs.append((i, i + 2))
    #         else:
    #             next_to_next_pairs.append((i + 2, i))

    #     # indices = list(range(n))
    #     # adjacent_pairs = [(i, j) for i in indices for j in indices if i == j + 1 or i == j - 1]
    #     # next_to_next_pairs = [(i, j) for i in indices for j in indices if i == j + 2 or i == j - 2]
        
    #     sampled_pairs = adjacent_pairs[:num_of_examples]  # take first N adjacent pairs
    #     m = len(sampled_pairs)
        
    #     if m < num_of_examples:
    #         remaining = num_of_examples - m
    #         extra_pairs = next_to_next_pairs[:remaining]
    #         sampled_pairs.extend(extra_pairs)
            
    #         if len(extra_pairs) < remaining:
    #             print(f"Hard difficulty: only {m + len(extra_pairs)} pairs available (adjacent + next_to_next) for entity type: {entity_type} and property: {property}")
    #             logging.info(f"Hard difficulty: only {m + len(extra_pairs)} pairs available (adjacent + next_to_next) for entity type: {entity_type} and property: {property}")


    # For easy difficulty, we don't restrict the pairs to be adjacent
    elif difficulty == "easy":
        perm = list(itertools.permutations(range(n), 2))
        try:
            sampled_pairs = random.sample(perm, num_of_examples)
        except ValueError:
            print(f"Easy difficulty: {num_of_examples} to sample from list of len {len(perm)} for entity type: {entity_type} and property: {property} containing number of entities: {n}")
            logging.info(f"Easy difficulty: {num_of_examples} to sample from list of len {len(perm)} for entity type: {entity_type} and property: {property} containing number of entities: {n}")
            sampled_pairs = perm
    
    selected_item_pairs = [(next_ranked_prop_entities.iloc[i], next_ranked_prop_entities.iloc[j]) for (i, j) in sampled_pairs]
    for pair in selected_item_pairs:
        entity1 = pair[0][entity_header]
        entity2 = pair[1][entity_header]
        if compare_entities(pair[0][property_header], pair[1][property_header], ascending):
            example = {
                "entity_type": entity_type,
                "entity1": entity1,
                "entity2": entity2,
                "relation": relation,
                "answer": "Yes",
                "property": property,
                "high_property": high_property,
                "low_property": low_property,
                "question": generate_question(question, entity1, entity2)
            }
        else:
            example = {
                "entity_type": entity_type,
                "entity1": entity1,
                "entity2": entity2,
                "relation": relation,
                "answer": "No",
                "property": property,
                "high_property": high_property,
                "low_property": low_property,
                "question": generate_question(question, entity1, entity2)
            }
        examples.append(example)
    return examples

def main(dataset, output_file, number_of_adjacent_pairs_per_dataset = 1000):
    with open(output_file, "w", encoding="utf-8") as f:
        for dat in dataset:
            data_files_metadata = get_data_files_declarative_bias([dat])
            n = len(data_files_metadata)
            if dat == 'physical':
                n = 3
            num_of_examples_for_adjacent_pairs = int(number_of_adjacent_pairs_per_dataset / n)
            for data_file in data_files_metadata:
                file_path = os.path.join(configuration.ranking_dataset_path, data_file["file_path"])
                entity_type = data_file["entity_type"]
                property_header = data_file["property_header"]
                entity_header = data_file["entity_header"]
                property = data_file["property"]
                high_property = data_file["high property"]
                low_property = high_property # low_property = high_property becuase we are not using low property in the dataset
                ascending = data_file["ascending"]

                if "question" not in data_file:
                    print(f"⚠️ Warning: 'question' key not found in the data file metadata for {file_path}.")

                question = data_file["question"]
                ranking_header = data_file["ranking_header"]
                relation = property_header
            
                df = read_data(file_path)

                if "selectAll" in data_file and data_file["selectAll"]:
                    top_entities= df
                else:
                    top_entities= df.sort_values(by=ranking_header, ascending=False).head(topPopular)

                logging.info(f"Top {topPopular} entities for {entity_type} based on {ranking_header} in {file_path}:")
                logging.info("==================================")
                logging.info(top_entities)
                logging.info('\n\n')

                converted = pd.to_numeric(top_entities[property_header], errors='coerce')
                num_invalid = converted.isna().sum()

                if num_invalid > 0:
                    print(f"⚠️ Warning: {num_invalid} non-numeric value(s) found in '{property_header}' column. They will be treated as NaN and ignored.")

                top_entities[property_header] = converted
                top_entities = top_entities.dropna(subset=[property_header])
            

                ranked_prop_entities = top_entities.drop_duplicates(subset=[property_header], keep='last').sort_values(by=property_header, ascending=ascending)

                logging.info(f"Ranked entities based on {property_header}:")
                logging.info("==================================")
                logging.info(ranked_prop_entities)
                logging.info('\n\n')

                examples = create_adjacency_examples(ranked_prop_entities, num_of_examples_for_adjacent_pairs, entity_type, relation, property, entity_header, property_header, ascending, high_property, low_property, question)
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            if dat == 'physical':
                examples = select_physical_properties_dataset(num_of_examples_for_adjacent_pairs)
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    initialization()
    main(train_dataset, train_output_file, num_train_examples_per_dataset)
    main(test_dataset, test_output_file, num_test_examples_per_dataset)