from process_data import (
    read_csv_and_json,
    transform_pubmeds,
    transform_clinical_trials,
    transform_drugs,
    build_graph_from_dataframes,
    generate_json_from_graph
)

def main():
    dict_df = read_csv_and_json(
        "../data/input/drugs.csv",
        "../data/input/pubmed.csv",
        "../data/input/clinical_trials.csv",
        "../data/input/pubmed.json"
    )
    dict_df = transform_pubmeds(dict_df)
    dict_df = transform_clinical_trials(dict_df)
    dict_df = transform_drugs(dict_df)
    merged_graph = build_graph_from_dataframes(dict_df)
    generate_json_from_graph(merged_graph)
    #TODO: Use a logger instead of print
    print("Med Pipeline Job Finished !")

if __name__ == "__main__":
    main()
