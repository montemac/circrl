conda deactivate
conda activate circrl
rm -rf dist/*
python3 -m build
twine upload dist/*