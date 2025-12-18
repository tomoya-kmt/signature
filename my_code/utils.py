import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, List
from joblib import Parallel, delayed

def convert_to_geodataframe(df, lon_col="lon", lat_col="lat", crs="EPSG:6668"):
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


def nearest_merge(
    gdf_a, gdf_b, geom_col_a="geometry", geom_col_b="geometry", threshold=None
):
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

    # gdf_b["geometry_b"] = gdf_b[geom_col_b]

    # 元のgeometry列を一時的に設定
    gdf_a = gdf_a.set_geometry(geom_col_a)
    gdf_b = gdf_b.set_geometry(geom_col_b)

    # 投影座標系に変換(日本測地系2011/UTM zone 54N)
    gdf_a_proj = gdf_a.to_crs("EPSG:6677")
    gdf_b_proj = gdf_b.to_crs("EPSG:6677")

    # 空間インデックスを使用して最近傍を検索
    nearest = gdf_a_proj.sjoin_nearest(
        gdf_b_proj, how="left", distance_col="distance", max_distance=threshold
    )

    # 元の座標系に戻す
    nearest = nearest.to_crs(gdf_a.crs)
    nearest = nearest.drop(columns=["index_right"])

    return nearest


def spatial_join(
    gdf_left,
    gdf_right,
    how="left",
    predicate="intersects",
    lsuffix="left",
    rsuffix="right",
):
    """
    geopandasのsjoinを用いて2つのGeoDataFrameを空間結合します。

    Parameters:
    gdf_left (gpd.GeoDataFrame): 左側のGeoDataFrame
    gdf_right (gpd.GeoDataFrame): 右側のGeoDataFrame
    how (str): 結合方法 ('left', 'right', 'inner')
    predicate (str): 空間述語 ('intersects', 'contains', 'within', 'crosses', 'overlaps', 'touches')
    lsuffix (str): 左側のカラム名が重複した場合のサフィックス
    rsuffix (str): 右側のカラム名が重複した場合のサフィックス

    Returns:
    gpd.GeoDataFrame: 空間結合されたGeoDataFrame
    """
    # CRSが一致しているか確認
    if gdf_left.crs != gdf_right.crs:
        print(f"Warning: CRSが異なります。gdf_right を {gdf_left.crs} に変換します。")
        gdf_right = gdf_right.to_crs(gdf_left.crs)

    # 空間結合を実行
    result = gpd.sjoin(
        gdf_left,
        gdf_right,
        how=how,
        predicate=predicate,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
    )

    return result

def _read_single_geojson(file_path: Path) -> Optional[gpd.GeoDataFrame]:
    """
    単一のGeoJSONファイルをpyogrioエンジンで高速に読み込むヘルパー関数。
    並列処理のワーカーとして動作します。

    Parameters
    ----------
    file_path : Path
        読み込むファイルのパス

    Returns
    -------
    Optional[gpd.GeoDataFrame]
        読み込み成功時はGeoDataFrame、失敗時はNone
    """
    try:
        # engine="pyogrio" を指定することで高速化
        return gpd.read_file(file_path, engine="pyogrio")
    except Exception as e:
        print(f"エラー: {file_path} の読み込みに失敗しました: {e}")
        return None

def merge_geojson_files(folder_path: str) -> Optional[gpd.GeoDataFrame]:
    """
    指定されたフォルダ内のすべてのgeojsonファイルを並列処理で高速に読み込み、
    1つのGeoDataFrameに結合します。

    Parameters
    ----------
    folder_path : str
        geojsonファイルが格納されているフォルダのパス

    Returns
    -------
    Optional[gpd.GeoDataFrame]
        結合されたGeoDataFrame。ファイルが見つからない場合はNone
    """
    folder = Path(folder_path)
    
    # .geojsonファイルを再帰的に検索
    geojson_files = list(folder.glob("**/*.geojson"))
    
    if not geojson_files:
        print(f"警告: {folder_path} にgeojsonファイルが見つかりませんでした。")
        return None
    
    print(f"{len(geojson_files)}個のgeojsonファイルを全CPUコアを使用して並列読み込みします...")
    
    # 【高速化ポイント】
    # Joblibを使用して、ファイル読み込みを並列実行 (n_jobs=-1 で全コア使用)
    gdfs: List[Optional[gpd.GeoDataFrame]] = Parallel(n_jobs=-1)(
        delayed(_read_single_geojson)(fp) for fp in geojson_files
    )
    
    # 読み込みに失敗したNoneを除去
    valid_gdfs = [gdf for gdf in gdfs if gdf is not None]
    
    if not valid_gdfs:
        print("警告: 有効なgeojsonファイルが読み込めませんでした。")
        return None
    
    # すべてのGeoDataFrameを結合
    # pandas.concatは非常に高速です
    merged_df = pd.concat(valid_gdfs, ignore_index=True)
    
    # pandas.concatの結果はDataFrameになることがあるため、GeoDataFrameに再変換
    # 最初の有効なデータのCRSとGeometryカラム名を継承
    first_gdf = valid_gdfs[0]
    merged_gdf = gpd.GeoDataFrame(merged_df, crs=first_gdf.crs, geometry=first_gdf.geometry.name)
    
    print(f"結合完了: 合計 {len(merged_gdf)} 件のフィーチャー")
    
    return merged_gdf