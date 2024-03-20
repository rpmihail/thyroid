
# Predict malignant/benign using contextual semantic interpretability

Use the cdoes in final_code directory under code/ directory. Steps for running the code:

1. Run the save_images.py to generate label_list_<split>.txt.txt file and a directory with the masked images.
2. Copy the file and the directory generated above and place under codes/ directory
3. Sankey plot is automatically generated and so are the CAMs. Please create a directory called heatmaps<expt_num> for the CAMs before executing code.
4. Execute either the one layer or the two layer type predictionnetwork using:
  ``` python End-to-end_1fc.py <expt_num>```
  or 
``` python End-to-end_2fc.py <expt_num>```
 
