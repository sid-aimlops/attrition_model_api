python3 -m venv venv 
source venv/bin/activate
cd requirements
pip install -r requirements.txt

Python doesn’t recognize the project structure as a package. The sys.path.append(str(root)) you have included may not be effectively adding the correct path in this context.

python -m attrition_model.config.corec

This ensures that Python treats your project as a package, making attrition_model accessible.

export PYTHONPATH="/Users/sid/Documents/AI-ML Course/Projects/Attrition Predictor:$PYTHONPATH"
python /Users/sid/Documents/AI-ML\ Course/Projects/Attrition\ Predictor/attrition_model/config/core.py