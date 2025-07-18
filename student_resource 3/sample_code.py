import os
import pandas as pd
from src.testing import modelmain  # Ensure modelmain is correctly imported from src.testing

def predictor(image_link, entity_name, index, group_id):
    '''
    Call your model/approach here
    '''
    
    # Create a temporary dataset with a single row
    temp_df = pd.DataFrame({
        'index': [index],
        'image_link': [image_link],
        'group_id': [group_id],  # Assuming you don't need it for a single prediction
        'entity_name': [entity_name]
    })

    # Process the temporary dataset
    results = modelmain(temp_df)
    
    # Get the result for the single image
    result = results[0]['prediction']
    return result

if __name__ == "__main__":
    # Define the dataset folder path relative to this script
    DATASET_FOLDER = 'student_resource 3/dataset'
    
    # Define the file paths
    test_file_path = os.path.join(DATASET_FOLDER, 'dg.csv')
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    
    # Read the test CSV file without headers
    test = pd.read_csv(test_file_path, header=None)
    
    # Assign column names manually based on the content
    test.columns = ['index', 'image_link', 'group_id', 'entity_name']
    
    # Print column names to verify correct column names
    print("Columns in CSV:", test.columns)
    
    # Filter rows based on index range
    filtered_test = test[(test['index'] >= 100001) & (test['index'] <= 131287)]
    
    # Apply the predictor function to each row in the filtered dataset
    filtered_test['prediction'] = filtered_test.apply(
        lambda row: predictor(row['image_link'], row['entity_name'], row['index'], row['group_id']), axis=1)
    
    # Save the predictions to a new CSV file
    filtered_test[['index', 'prediction']].to_csv(output_filename, index=False)
