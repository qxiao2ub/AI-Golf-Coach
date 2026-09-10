.PHONY: install run test check

install:
	python -m pip install -r requirements.txt

run:
	streamlit run streamlit_app.py

test:
	python -m pytest -q

check:
	python -m compileall -q streamlit_app.py golf_coach tests
