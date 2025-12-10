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


def nearest_merge(gdf_a, gdf_b, geom_col_a='geometry', geom_col_b='geometry', threshold=None):
    """
    GeoDataFrame AのそれぞれのジオメトリーについてGeoDataFrame Bの最も近いジオメトリーをマージする。
    距離が閾値以上の場合はBの列をNULLでマージ。
    
    Parameters:
    gdf_a (gpd.GeoDataFrame): ベースとなるGeoDataFrame
    gdf_b (gpd.GeoDataFrame): マージ元のGeoDataFrame
    geom_col_a (str): gdf_aのジオメトリー列名
    geom_col_b (str): gdf_bのジオメトリー列名
    threshold (float): 距離の閾値。この値以上の距離の場合はマッチ無しとする
    
    Returns:
    gpd.GeoDataFrame: マージされたGeoDataFrame
    """
    # 元のgeometry列を一時的に設定
    gdf_a = gdf_a.set_geometry(geom_col_a)
    gdf_b = gdf_b.set_geometry(geom_col_b)
    
    # 空間インデックスを使用して最近傍を検索
    nearest = gdf_a.sjoin_nearest(gdf_b, how='left', distance_col='_distance', max_distance=threshold)
    
    # 重複する列名を処理(index_rightを除外し、Bのカラムのみを保持)
    b_cols = [col for col in gdf_b.columns if col != geom_col_b]
    result_cols = list(gdf_a.columns) + b_cols + ['_distance']
    
    # index_rightを使ってBのデータをマージ
    if 'index_right' in nearest.columns:
        nearest = nearest.merge(gdf_b[b_cols], left_on='index_right', right_index=True, how='left', suffixes=('', '_b'))
        nearest = nearest.drop(columns=['index_right'])
    
    return nearest