import pandas as pd
from g2_hurdle.utils.io import load_data

def test_load_data_splits_store_menu(tmp_path):
    df = pd.DataFrame({
        "영업일자": pd.to_datetime(["2024-01-01"]),
        "매출수량": [1],
        "영업장명_메뉴명": ["s1_m1"],
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    cfg = {
        "data": {
            "date_col_candidates": ["영업일자"],
            "target_col_candidates": ["매출수량"],
            "id_col_candidates": ["영업장명_메뉴명", "store_id", "menu_id"],
        },
    }
    out, schema = load_data(str(csv_path), cfg)
    assert out.loc[0, "store_id"] == "s1"
    assert out.loc[0, "menu_id"] == "m1"
    assert "store_id" in schema["series"]
    assert "menu_id" in schema["series"]
