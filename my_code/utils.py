import pandas as pd
import geopandas as gpd

def convert_to_geodataframe(df, lon_col='lon', lat_col='lat', crs='EPSG:4326'):
    """
    経度と緯度の列を持つpandas DataFrameをGeoDataFrameに変換します。

    Parameters:
    df (pd.DataFrame): 経度と緯度の列を含む入力DataFrame
    lon_col (str): 経度の値を含む列名
    lat_col (str): 緯度の値を含む列名
    crs (str): GeoDataFrameの座標参照系

    Returns:
    gpd.GeoDataFrame: 経度と緯度から作成されたgeometry列を持つGeoDataFrame
    """
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf

def drop_cols(df, drop_cols=[], threshold=1):
    # 欠損値の割合が90%以上のカラムを削除
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio >= threshold].index.tolist()
    
    # 削除するカラムと欠損率を表示
    if cols_to_drop:
        print("削除するカラムと欠損率:")
        for col in cols_to_drop:
            print(f"  {col}: {missing_ratio[col]:.2%}")
    else:
        print("削除するカラムはありません")
    
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=drop_cols)
    
    return df