web: mkdir -p data models && ([ -f models/energy_model.pkl ] || python -m src.train) && python -m src.db_check && streamlit run app.py --server.port $PORT --server.address 0.0.0.0
