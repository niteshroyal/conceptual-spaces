import re
from conf import configuration


def get_score(score1, score2):
    if score1 > score2:
        answer = 'Yes'
    else:
        answer = 'No'
    return answer


def get_score_inverse(score1, score2):
    if score1 < score2:
        answer = 'Yes'
    else:
        answer = 'No'
    return answer


def get_prompt_taste(record1, record2, filename):
    datapoints = []
    if filename == 'food_Taste.txt':
        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sweet_Mean'])
        score2 = float(record2['Sweet_Mean'])
        question = f'This question is about two food items: Is {element1} generally sweeter in taste than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Salty_Mean'])
        score2 = float(record2['Salty_Mean'])
        question = f'This question is about two food items: Is {element1} generally saltier than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Sour_Mean'])
        score2 = float(record2['Sour_Mean'])
        question = f'This question is about two food items: Is {element1} generally more sour in taste than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Bitter_Mean'])
        score2 = float(record2['Bitter_Mean'])
        question = f'This question is about two food items: Is {element1} generally more bitter in taste than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Umami_Mean'])
        score2 = float(record2['Umami_Mean'])
        question = f'This question is about two food items: Is {element1} generally more umami than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['foodLabel']
        element2 = record2['foodLabel']
        score1 = float(record1['Fat_Mean'])
        score2 = float(record2['Fat_Mean'])
        question = f'This question is about two food items: Does {element1} taste fattier than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints


def get_prompt_movies(record1, record2, filename):
    datapoint = dict()
    pattern = r"NEW(.+?)_movie_titles1\.txt"
    if re.search(pattern, filename):
        keyword = re.search(pattern, filename).group(1).lower()
        element1 = record1['title']
        element2 = record2['title']
        score1 = float(record1['score'])
        score2 = float(record2['score'])
        question = f'This question is about two movies: Is {element1} more {keyword} than {element2}?'
        answer = get_score(score1, score2)
    else:
        raise Exception('Filename mismatch.')
    datapoint['question'] = question
    datapoint['answer'] = answer
    return [datapoint]


def get_prompt_books(record1, record2, filename):
    datapoint = dict()
    pattern = r"NEW_(.+?)_Book_titles1\.txt"
    if re.search(pattern, filename):
        keyword = re.search(pattern, filename).group(1).lower()
        element1 = record1['title']
        element2 = record2['title']
        score1 = float(record1['score'])
        score2 = float(record2['score'])
        question = f'This question is about two books: Is {element1} more {keyword} than {element2}?'
        answer = get_score(score1, score2)
    else:
        raise Exception('Filename mismatch.')
    datapoint['question'] = question
    datapoint['answer'] = answer
    return [datapoint]


def get_prompt_rocks(record1, record2, filename):
    datapoints = []
    if filename == 'rock_data.txt':
        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['lightness'])
        score2 = float(record2['lightness'])
        question = f'This question is about two types of rocks: Is {element1} lighter in color than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['grainSize'])
        score2 = float(record2['grainSize'])
        question = f'This question is about two types of rocks: Is {element1} more coarse than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['roughness'])
        score2 = float(record2['roughness'])
        question = f'This question is about two types of rocks: Is {element1} rougher than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['shine'])
        score2 = float(record2['shine'])
        question = f'This question is about two types of rocks: Is {element1} more shiny than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['organization'])
        score2 = float(record2['organization'])
        question = f'This question is about two types of rocks: Does {element1} have a more uniform grain structure than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['variability'])
        score2 = float(record2['variability'])
        question = f'This question is about two types of rocks: Does {element1} have more variability in color than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)

        element1 = record1['rockLabel']
        element2 = record2['rockLabel']
        score1 = float(record1['density'])
        score2 = float(record2['density'])
        question = f'This question is about two types of rocks: Is {element1} denser than {element2}?'
        answer = get_score(score1, score2)
        datapoint = dict()
        datapoint['question'] = question
        datapoint['answer'] = answer
        datapoints.append(datapoint)
    else:
        raise Exception('Filename mismatch.')
    return datapoints


def get_wikidata_prompts(record1, record2, filename):
    if filename == 'matched_outputRiver_WikiPageRank.txt':
        element1 = record1['riverLabel']
        element2 = record2['riverLabel']
        score1 = float(record1['length'])
        score2 = float(record2['length'])
        question = f'This question is about two rivers: Is {element1} longer than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_cityRankedPopulation_WikiRank.txt':
        element1 = record1['cityLabel']
        element2 = record2['cityLabel']
        score1 = float(record1['maxPopulation'])
        score2 = float(record2['maxPopulation'])
        question = f'This question is about two cities: Does {element1} have a larger population than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_PersonBorn_London_WikiRank.txt':
        element1 = record1['personLabel']
        element2 = record2['personLabel']
        score1 = float(record1['dob'])
        score2 = float(record2['dob'])
        question = f'This question is about two persons: Was {element1} born after {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_Rank_BuildingHeight_WikiRank.txt':
        element1 = record1['itemLabel']
        element2 = record2['itemLabel']
        score1 = float(record1['maxHeight'])
        score2 = float(record2['maxHeight'])
        question = f'This question is about two buildings: Is {element1} taller than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_Rank_Island_Area_WikiRank.txt':
        element1 = record1['islandLabel']
        element2 = record2['islandLabel']
        score1 = float(record1['No_islandArea'])
        score2 = float(record2['No_islandArea'])
        question = f'This question is about two islands: Is {element1} larger than {element2} in area?'
        answer = get_score(score1, score2)
    elif filename == 'matched_RankMusueumsLattitude_Italy_WikiRank.txt':
        element1 = record1['museumLabel']
        element2 = record2['museumLabel']
        score1 = float(record1['Rank_lat'])
        score2 = float(record2['Rank_lat'])
        question = f'This question is about two museums in Italy: Is {element1} located at a higher latitude compared to {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'unique_matched_InceptionCompanyWikiPageRank.txt':
        element1 = record1['companyLabel']
        element2 = record2['companyLabel']
        score1 = float(record1['minInception'])
        score2 = float(record2['minInception'])
        question = f'This question is about two companies: Was {element1} founded after {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'unique_matched_MountainHeightWikiPageRank.txt':
        element1 = record1['mountainLabel']
        element2 = record2['mountainLabel']
        score1 = float(record1['elevation'])
        score2 = float(record2['elevation'])
        question = f'This question is about two mountains: Does {element1} have a higher elevation than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'unique_matched_Person_SOcialMedia_WikiPageRank.txt':
        element1 = record1['personLabel']
        element2 = record2['personLabel']
        score1 = float(record1['maxSocialMediaFollower'])
        score2 = float(record2['maxSocialMediaFollower'])
        question = f'This question is about two persons: Does {element1} have more social media followers than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'unique_matched_speciesMass_WikiPageRank.txt':
        element1 = record1['speciesLabel']
        element2 = record2['speciesLabel']
        score1 = float(record1['maxMass'])
        score2 = float(record2['maxMass'])
        question = f'This question is about two species: Is {element1} generally heavier than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_queryAcademyAward_Direction_WikiRank.txt':
        element1 = record1['directorLabel']
        element2 = record2['directorLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two directors: Has {element1} won more Academy Awards than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_awardActor_WikiRank.txt':
        element1 = record1['actorLabel']
        element2 = record2['actorLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two actors: Has {element1} won more Academy Awards for Best Actor than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_chemcialELements_DIscovery_WikiPageRank.txt':
        element1 = record1['elementLabel']
        element2 = record2['elementLabel']
        score1 = float(record1['minDiscovery'])
        score2 = float(record2['minDiscovery'])
        question = f'This question is about two chemical elements: Was {element1} discovered before {element2}?'
        answer = get_score_inverse(score1, score2)
    elif filename == 'matched_queryfood_WaterFootPrint_WikiRank.txt':
        element1 = record1['foodGrpLabel']
        element2 = record2['foodGrpLabel']
        score1 = float(record1['WaterFootPrint'])
        score2 = float(record2['WaterFootPrint'])
        question = f'This question is about two types of food: Does {element1} have a larger water footprint than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_noGrammyAward_Composer_WikiRank.txt':
        element1 = record1['artistLabel']
        element2 = record2['artistLabel']
        score1 = float(record1['numAwards'])
        score2 = float(record2['numAwards'])
        question = f'This question is about two artists: Has {element1} received more awards than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_rankCountries_Population.txt':
        element1 = record1['countryLabel']
        element2 = record2['countryLabel']
        score1 = float(record1['maxPopulation'])
        score2 = float(record2['maxPopulation'])
        question = f'This question is about two countries: Does {element1} have a larger population than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_rankElements_AtomicNo_WikiPageRank.txt':
        element1 = record1['elementLabel']
        element2 = record2['elementLabel']
        score1 = float(record1['atomicNo'])
        score2 = float(record2['atomicNo'])
        question = f'This question is about two chemical elements: Does {element1} have a higher atomic number than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_query_schoville_WikiRank.txt':
        element1 = record1['foodName']
        element2 = record2['foodName']
        score1 = float(record1['Rank_scovilleGrade'])
        score2 = float(record2['Rank_scovilleGrade'])
        question = f'This question is about two types of food: Does {element1} have a higher Scoville grade than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_RankBuildingElevators_WikiRank.txt':
        element1 = record1['buildingLabel']
        element2 = record2['buildingLabel']
        score1 = float(record1['no_elevator'])
        score2 = float(record2['no_elevator'])
        question = f'This question is about two buildings: Does {element1} have more elevators than {element2}?'
        answer = get_score(score1, score2)
    elif filename == 'matched_rankMusicalObjects_ByInceptionDate.txt':
        element1 = record1['instrumentLabel']
        element2 = record2['instrumentLabel']
        score1 = float(record1['inceptDate'])
        score2 = float(record2['inceptDate'])
        question = f'This question is about two instruments: Did {element1} exist before {element2}?'
        answer = get_score_inverse(score1, score2)

    elif filename == 'food_Taste.txt':
        if configuration.entity_type == 'taste_sweet':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Sweet_Mean'])
            score2 = float(record2['Sweet_Mean'])
            question = f'This question is about two food items: Is {element1} generally sweeter in taste than {element2}?'
            # question = f'This question is about two food items: Is {element1} sweeter in taste than {element2}?'
            # question = f'This question is about two food items: Should we rank {element1} higher than {element2} in terms of sweet taste?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'taste_salty':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Salty_Mean'])
            score2 = float(record2['Salty_Mean'])
            question = f'This question is about two food items: Is {element1} generally saltier than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'taste_sour':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Sour_Mean'])
            score2 = float(record2['Sour_Mean'])
            question = f'This question is about two food items: Is {element1} generally more sour in taste than {element2}?'
            # question = f'This question is about two food items: Should we rank {element1} higher than {element2} in terms of sour taste?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'taste_bitter':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Bitter_Mean'])
            score2 = float(record2['Bitter_Mean'])
            question = f'This question is about two food items: Is {element1} generally more bitter in taste than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'taste_umami':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Umami_Mean'])
            score2 = float(record2['Umami_Mean'])
            question = f'This question is about two food items: Is {element1} generally more umami than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'taste_fat':
            element1 = record1['foodLabel']
            element2 = record2['foodLabel']
            score1 = float(record1['Fat_Mean'])
            score2 = float(record2['Fat_Mean'])
            question = f'This question is about two food items: Does {element1} taste fattier than {element2}?'
            answer = get_score(score1, score2)
        else:
            raise Exception('Unhandled')
    elif filename == 'rock_data.txt':
        if configuration.entity_type == 'rocks_lightness':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['lightness'])
            score2 = float(record2['lightness'])
            question = f'This question is about two types of rocks: Is {element1} lighter in color than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_grainSize':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['grainSize'])
            score2 = float(record2['grainSize'])
            question = f'This question is about two types of rocks: Is {element1} more coarse than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_roughness':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['roughness'])
            score2 = float(record2['roughness'])
            question = f'This question is about two types of rocks: Is {element1} rougher than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_shine':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['shine'])
            score2 = float(record2['shine'])
            question = f'This question is about two types of rocks: Is {element1} more shiny than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_organization':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['organization'])
            score2 = float(record2['organization'])
            question = f'This question is about two types of rocks: Does {element1} have a more uniform grain structure than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_variability':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['variability'])
            score2 = float(record2['variability'])
            question = f'This question is about two types of rocks: Does {element1} have more variability in color than {element2}?'
            answer = get_score(score1, score2)
        elif configuration.entity_type == 'rocks_density':
            element1 = record1['rockLabel']
            element2 = record2['rockLabel']
            score1 = float(record1['density'])
            score2 = float(record2['density'])
            question = f'This question is about two types of rocks: Is {element1} denser than {element2}?'
            answer = get_score(score1, score2)
        else:
            raise Exception('Unhandled')
    else:
        raise Exception('Dataset label mismatch.')
    datapoint = dict()
    datapoint['question'] = question
    datapoint['answer'] = answer
    return datapoint
