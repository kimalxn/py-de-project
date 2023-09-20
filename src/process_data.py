import pandas as pd
import json
from io import StringIO, BytesIO
import numpy as np
import networkx as nx

# Debugging options
#import matplotlib.pyplot as plt
#pd.options.display.max_colwidth = 200

# pubmed.json is not correctly formatted initially (invalid JSON file)

# Extract
def read_csv_and_json(drugs_file, pubmed_file, clinical_file, pubmed_json_file):
    df_drugs = pd.read_csv(drugs_file, delimiter=',')
    df_pubmed = pd.read_csv(pubmed_file, delimiter=',')
    df_clinical_trials = pd.read_csv(clinical_file, delimiter=',')
    df_pubmed_json = pd.read_json(pubmed_json_file)

    dict_df = {
    'df_drugs': df_drugs,
    'df_pubmed': df_pubmed,
    'df_clinical_trials': df_clinical_trials,
    'df_pubmed_json': df_pubmed_json
    }
    return dict_df

# Transform pubmed.csv and pubmed.json
def transform_pubmeds(dict_df):
    df_pubmed = dict_df['df_pubmed']
    df_pubmed_json = dict_df['df_pubmed_json']
    # Conversion de la date du pubmed.csv en format datetime64
    df_pubmed['date'] = pd.to_datetime(df_pubmed['date'], dayfirst=True)
    # Conversion de l'id du pubmed.json en float64
    df_pubmed_json['id'] = pd.to_numeric(df_pubmed_json['id'])
    # Homemade fillna pour donner un ID aux pubmed sans ID
    for i in range(len(df_pubmed_json)):
        if pd.isna(df_pubmed_json['id'][i]):
            df_pubmed_json.at[i, 'id'] = df_pubmed_json['id'][i - 1] + 1
    # Conversion de l'id de pubmed.json en int64
    df_pubmed_json['id'] = df_pubmed_json['id'].astype(np.int64)
    # Créé un DataFrame combinant les deux fichiers pubmed.csv and pubmed.json
    df_pubmerged = pd.merge(df_pubmed, df_pubmed_json, how='outer')
    dict_df['df_pubmerged'] = df_pubmerged
    return dict_df

# Transform clinical_trials.csv
def transform_clinical_trials(dict_df):
    df_clinical_trials = dict_df['df_clinical_trials']
    # Fusionner les lignes avec le même titre
    df_clinical_trials_merged = df_clinical_trials.groupby('scientific_title', group_keys=False).apply(lambda x: x.ffill().bfill())
    # Supprimer les doublons de titre
    df_clinical_trials_merged.drop_duplicates(subset='scientific_title', keep='first', inplace=True)
    # Suppression de la publication scientifique NCT04237091 qui n'existe pas
    df_clinical_trials_merged = df_clinical_trials_merged[df_clinical_trials_merged.id != "NCT04237091"]
    # Retrait des caractères ASCII
    # TODO: Réaliser une moulinette qui retire automatiquement les caractères ASCII non formatés plutôt que de le réaliser un par un
    df_clinical_trials_merged.at[7, 'journal'] = "Journal of emergency nursing"
    df_clinical_trials_merged.at[2, 'scientific_title'] = "Feasibility of a Clinical Trial Comparing the Use of Cetirizine to Replace Diphenhydramine in the Prevention of Reactions Related to Paclitaxel"
    # Insersion du nom de l'article scientifique NCT04237090, non renseigné
    df_clinical_trials_merged.at[4, 'scientific_title'] = "Preemptive Infiltration With Betamethasone and Ropivacaine for Postoperative Pain in Laminoplasty or Laminectomy"
    # Conversion de la date en format datetime64
    df_clinical_trials_merged['date'] = pd.to_datetime(df_clinical_trials_merged['date'], dayfirst=True)
    # Réinitialisation de l'index
    df_clinical_trials_merged.reset_index(drop=True, inplace=True)
    dict_df['df_clinical_trials_merged'] = df_clinical_trials_merged
    return dict_df

def transform_drugs(dict_df):
    df_drugs = dict_df['df_drugs']
    df_pubmerged = dict_df['df_pubmerged']
    df_clinical_trials_merged = dict_df['df_clinical_trials_merged']
    # Creation de nouvelles colonnes avec df_drugs en base pour établir le DAG final
    df_drugs_to_json = df_drugs
    df_drugs_to_json['pubmed_mentionned_id'] = np.empty((len(df_drugs_to_json), 0)).tolist()
    df_drugs_to_json['pubmed_mentionned_date'] = np.empty((len(df_drugs_to_json), 0)).tolist()
    df_drugs_to_json['scientific_title_mentionned_id'] = np.empty((len(df_drugs_to_json), 0)).tolist()
    df_drugs_to_json['scientific_title_mentionned_date'] = np.empty((len(df_drugs_to_json), 0)).tolist()
    df_drugs_to_json['journal_mentionned'] = np.empty((len(df_drugs_to_json), 0)).tolist()
    df_drugs_to_json['journal_mentionned_date'] = np.empty((len(df_drugs_to_json), 0)).tolist()

    # Peuplement du dataframe df_drugs_to_json, qui représente les mentions de chaque médicament de diverses sources
    journal_entries = []

    for index, drug in enumerate(df_drugs_to_json['drug']):
        for p_title in df_pubmerged['title']:
            if drug.lower() in p_title.lower():
                df_drugs_to_json.at[index, 'pubmed_mentionned_id'].append(
                    df_pubmerged.loc[df_pubmerged['title'] == p_title, 'id'].values[0]
                )
                df_drugs_to_json.at[index, 'pubmed_mentionned_date'].append(
                    df_pubmerged.loc[df_pubmerged['title'] == p_title, 'date'].values[0]
                )
                journal_entries.append(
                    (
                        df_pubmerged.loc[df_pubmerged['title'] == p_title, 'journal'].values[0],
                        df_pubmerged.loc[df_pubmerged['title'] == p_title, 'date'].values[0]
                    )
                )
        for s_title in df_clinical_trials_merged['scientific_title']:
            if drug.lower() in s_title.lower():
                df_drugs_to_json.at[index, 'scientific_title_mentionned_id'].append(
                    df_clinical_trials_merged.loc[df_clinical_trials_merged['scientific_title'] == s_title, 'id'].values[0]
                )
                df_drugs_to_json.at[index, 'scientific_title_mentionned_date'].append(
                    df_clinical_trials_merged.loc[df_clinical_trials_merged['scientific_title'] == s_title, 'date'].values[0]
                )
                journal_entries.append(
                    (
                        df_clinical_trials_merged.loc[df_clinical_trials_merged['scientific_title'] == s_title, 'journal'].values[0],
                        df_clinical_trials_merged.loc[df_clinical_trials_merged['scientific_title'] == s_title, 'date'].values[0]
                    )
                )
        # Supprimes les tuples en double dans les entrées du journal (i.e. même journal / même jour de publication)
        journal_entries = list(set(journal_entries))
        while journal_entries:
            entry = journal_entries.pop()
            df_drugs_to_json.at[index, 'journal_mentionned'].append(entry[0])
            df_drugs_to_json.at[index, 'journal_mentionned_date'].append(entry[1])

    # TODO: retirer les cellules comprenant [] et les remplacer par np.nan
    dict_df['df_drugs_to_json'] = df_drugs_to_json
    return dict_df


# Création d'un graphe pour chaque médicament avec NetworkX.
# Chaque noeud représente soit un médicament, soit une publication PubMed, soit un journal, soit un Clinical trial.
# Les liaisons représentent les références des médicaments dans ses mentions. Ces liaisons portent la date de la mention.

# Chaque graphe (un par drug) sera mergé dans un unique graphe
def build_graph_from_dataframes(dict_df):
    df_drugs_to_json = dict_df['df_drugs_to_json']
    df_pubmerged = dict_df['df_pubmerged']
    df_clinical_trials_merged = dict_df['df_clinical_trials_merged']

    A = nx.MultiDiGraph()

    for drug_index, drug in enumerate(df_drugs_to_json['drug']):
        G = nx.MultiDiGraph()
        G.add_node(df_drugs_to_json["drug"][drug_index], label="drug", atccode=df_drugs_to_json["atccode"][drug_index])

        list_pubmed_mention = df_drugs_to_json["pubmed_mentionned_id"][drug_index]
        list_pubmed_date = df_drugs_to_json["pubmed_mentionned_date"][drug_index]

        for index, id in enumerate(list_pubmed_mention):
            G.add_node(
                list_pubmed_mention[index],
                title=df_pubmerged.loc[df_pubmerged['id'] == id, 'title'].values[0],
                label="pubmed"
            )
            G.add_edge(
                df_drugs_to_json["drug"][drug_index],
                list_pubmed_mention[index],
                date_mention=list_pubmed_date[index],
                label="pubmed_mentionned"
            )

        list_scientific_title_mention = df_drugs_to_json["scientific_title_mentionned_id"][drug_index]
        list_scientific_title_mention_date = df_drugs_to_json["scientific_title_mentionned_date"][drug_index]

        for index, id in enumerate(list_scientific_title_mention):
            G.add_node(
                list_scientific_title_mention[index],
                title=df_clinical_trials_merged.loc[df_clinical_trials_merged['id'] == id, 'scientific_title'].values[0],
                label="Clinical trials")
            G.add_edge(
                df_drugs_to_json["drug"][drug_index],
                list_scientific_title_mention[index],
                date_mention=list_scientific_title_mention_date[index],
                label="clinical_trials_mentionned"
            )
        list_journal_mention = df_drugs_to_json["journal_mentionned"][drug_index]
        list_journal_mention_date = df_drugs_to_json["journal_mentionned_date"][drug_index]

        for index, id in enumerate(list_journal_mention):
            G.add_node(list_journal_mention[index], label="Journal")
            G.add_edge(
                df_drugs_to_json["drug"][drug_index],
                list_journal_mention[index],
                date_mention=list_journal_mention_date[index],
                label="journal_mentionned"
            )
        #nx.draw_networkx(G)
        A = nx.compose(A,G)
    return A

# Load
def generate_json_from_graph(graph):
    data1 = nx.node_link_data(graph)
    # Écriture dans le fichier JSON
    json_file = open('../data/output/graph.json', 'w')
    json.dump(data1, json_file, sort_keys = True, indent = 4, separators = (',', ': '), default=str, ensure_ascii=False)
    json_file.close()
    return True
