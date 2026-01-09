import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, List
from joblib import Parallel, delayed
import re

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

    # index_right列が既に存在する場合は削除
    if "index_right" in gdf_a.columns:
        gdf_a = gdf_a.drop(columns=["index_right"])
    if "index_right" in gdf_b.columns:
        gdf_b = gdf_b.drop(columns=["index_right"])

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
    # 左のデータフレームに一時的なID列を追加
    gdf_left_copy = gdf_left.copy()
    gdf_left_copy['_temp_id'] = range(len(gdf_left_copy))
    
    # CRSが一致しているか確認
    if gdf_left_copy.crs != gdf_right.crs:
        print(f"Warning: CRSが異なります。gdf_right を {gdf_left_copy.crs} に変換します。")
        gdf_right = gdf_right.to_crs(gdf_left_copy.crs)

    # 空間結合を実行
    result = gpd.sjoin(
        gdf_left_copy,
        gdf_right,
        how=how,
        predicate=predicate,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
    )
    
    # 一時ID列で重複を削除（最初の行を保持）
    result = result.drop_duplicates(subset=['_temp_id'], keep='first')
    
    # 一時ID列を削除
    result = result.drop(columns=['_temp_id'])

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

def merge_lag_features(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    join_key: str,
    prefix: str,
    col_year_map: dict[str, int]
) -> pd.DataFrame:
    """
    データフレームAに対し、データフレームB（横持ち）から過去5年分のラグ特徴量を結合する。

    データフレームBは事前に指定されたカラム名と年のマッピング辞書に基づいて
    縦持ち（Long format）に変換され、その後結合される。

    Parameters
    ----------
    df_a : pd.DataFrame
        ベースとなるデータフレーム。'target_ym'列（YYYYMM形式の整数または文字列、あるいはdatetime）を含む必要がある。
    df_b : pd.DataFrame
        年ごとのデータが横持ちで保存されているデータフレーム。
    join_key : str
        df_a と df_b を結合するためのキーとなるカラム名（例: ユーザーIDや地域コードなど）。
    col_year_map : dict[str, int]
        df_bのカラム名と、それが表す西暦年の対応辞書。
        例: {'L02_050': 2020, 'L02_051': 2021, ...}

    Returns
    -------
    pd.DataFrame
        過去5年分のラグ特徴量（lag_1_val ~ lag_5_val）が結合された新しいデータフレーム。

    """
    # データのコピー（元のデータフレームに影響を与えないため）
    df_result = df_a.copy()
    
    # 1. df_aのtarget_ymから「年」を抽出するための処理
    # target_ymがdatetime型の場合
    if pd.api.types.is_datetime64_any_dtype(df_result['target_ym']):
        df_result['_current_year'] = df_result['target_ym'].dt.year
    # target_ymが数値（202304など）や文字列の場合
    else:
        # 文字列に変換してから先頭4文字を取得して数値化
        df_result['_current_year'] = df_result['target_ym'].astype(str).str[:4].astype(int)

    # 2. df_bを縦持ち（Long format）に変換
    # mapに含まれるカラムのみを抽出してmeltする
    target_cols = list(col_year_map.keys())
    
    # id_varsにjoin_keyを指定し、value_varsに年のカラムを指定
    df_b_long = df_b.melt(
        id_vars=[join_key],
        value_vars=target_cols,
        var_name='_col_name',
        value_name='_value'
    )
    
    # カラム名を実際の「年」に変換
    df_b_long['_data_year'] = df_b_long['_col_name'].map(col_year_map)
    
    # 結合用に不要なカラムを削除
    df_b_prepared = df_b_long[[join_key, '_data_year', '_value']]

    # 3. 過去5年分のデータをループで結合
    for i in range(1, 6):
        lag_year_col = f'_join_year_lag_{i}'
        feature_name = f'lag_{i}_val_{prefix}'
        
        # 結合するための基準年（現在の年 - i年）を計算
        df_result[lag_year_col] = df_result['_current_year'] - i
        
        # マージ処理
        # 左：df_result (キー: join_key, lag_year_col)
        # 右：df_b_prepared (キー: join_key, _data_year)
        df_result = pd.merge(
            df_result,
            df_b_prepared,
            left_on=[join_key, lag_year_col],
            right_on=[join_key, '_data_year'],
            how='left'
        )
        
        # カラム名の整理
        df_result = df_result.rename(columns={'_value': feature_name})
        
        # マージに使った一時的なカラムを削除（_data_yearは右側のキーなのでマージ後に残る場合があるため削除）
        if '_data_year' in df_result.columns:
            df_result = df_result.drop(columns=['_data_year'])
        df_result = df_result.drop(columns=[lag_year_col])

    # 最終的な後処理（計算に使った一時カラムの削除）
    df_result = df_result.drop(columns=['_current_year'])
    
    return df_result


def generate_year_mapping(
    base_year: int,
    base_col_name: str,
    num_years: int,
    start_offset: int = -10
) -> dict[str, int]:
    """
    基準となる年とカラム名から、カラム名と西暦年のマッピング辞書を生成する。
    
    カラム名の末尾3桁を数値として扱い、基準年からの増減に合わせて西暦を割り当てる。
    前回のラグ特徴量生成（過去データの参照）に対応するため、デフォルトでは
    基準年の10年前からマッピングを開始する設定としている。

    Parameters
    ----------
    base_year : int
        基準となる西暦年（例: 2020）。
    base_col_name : str
        基準年におけるカラム名（例: 'L02_050'）。末尾が3桁の数字であることを前提とする。
    num_years : int
        生成するマッピングの総年数（カラム数）。
    start_offset : int, optional
        基準年から何年離れた地点から生成を開始するか（デフォルトは -10）。
        -10の場合、基準年の10年前から生成を開始する。

    Returns
    -------
    dict[str, int]
        {'カラム名': 西暦年} の形式の辞書。
        例: {'L02_040': 2010, ..., 'L02_050': 2020, ...}

    Raises
    ------
    ValueError
        base_col_nameの末尾が3桁の数字でない場合に発生。

    """
    # 正規表現で末尾の3桁の数字とそれ以前のプレフィックスを分離
    match = re.search(r'^(.*)(\d{3})$', base_col_name)
    if not match:
        raise ValueError(f"カラム名 '{base_col_name}' の末尾に3桁の数字が見つかりません。")

    prefix = match.group(1)      # 例: "L02_"
    base_suffix = int(match.group(2)) # 例: 50

    mapping = {}

    # start_offset から num_years 分だけループを回す
    # 例: offset=-10, num=20 の場合、基準-10年 ～ 基準+9年 までを生成
    for i in range(start_offset, start_offset + num_years):
        # 年の計算
        target_year = base_year + i
        
        # 末尾3桁の数字を計算
        target_suffix = base_suffix + i
        
        # 3桁より小さくなる（負になる）場合は考慮外とするか、データの仕様に合わせてエラー処理が必要
        # ここでは単純に0埋め3桁としてフォーマットする
        if target_suffix < 0:
            continue # 負のサフィックスは生成しない
            
        col_name = f"{prefix}{target_suffix:03}"
        mapping[col_name] = target_year

    return mapping

def join_nearest_with_conditions(
    gdf_base: gpd.GeoDataFrame,
    gdf_target: gpd.GeoDataFrame,
    match_col: str,
    max_dist: float,
    distance_col_name: str = "distance",
    target_crs: str = "EPSG:6677"
) -> gpd.GeoDataFrame:
    """
    指定されたカラムの値が一致するグループ内で、距離が最も近く、かつ閾値以内のジオメトリを結合します。
    ベースデータの全行を維持し、条件に合致するデータがない場合はNULLを埋めます(Left Join)。

    Parameters
    ----------
    gdf_base : gpd.GeoDataFrame
        ベースとなるデータフレーム（Geometry: Point想定）。
    gdf_target : gpd.GeoDataFrame
        結合対象のデータフレーム（Geometry: MultiLineString想定）。
    match_col : str
        両方のデータフレームに存在し、結合条件（等価）となるカラム名。
    max_dist : float
        結合を許可する最大距離。target_crsで指定した単位（メートル等）になります。
    target_crs : Optional[Union[str, int]], default None
        距離計算を行うための投影座標系（例: "EPSG:6677"）。
        指定された場合、結合前にこの座標系へ変換します。
    distance_col_name : str, default "distance"
        結果に含まれる距離情報のカラム名。

    Returns
    -------
    gpd.GeoDataFrame
        ベースデータの行数が維持されたデータフレーム。
        結合条件を満たさない行の結合先カラムにはNaNが入ります。
        ベース1行につきターゲット最大1行（m:1）となります。
    """
    
    # --- 1. 前処理: CRS変換とID付与 ---
    # 元データを変更しないようコピー
    base_process = gdf_base.copy()
    target_process = gdf_target.copy()

    # CRS変換
    if target_crs is not None:
        if base_process.crs is None or target_process.crs is None:
            raise ValueError("CRS変換を行うには、入力データフレームにCRSが定義されている必要があります。")
        base_process = base_process.to_crs(target_crs)
        target_process = target_process.to_crs(target_crs)
    elif base_process.crs != target_process.crs:
        target_process = target_process.to_crs(base_process.crs)

    # 確実にマージするために一時的なユニークIDを付与
    temp_id_col = "_temp_join_id"
    base_process[temp_id_col] = range(len(base_process))
    
    # --- 2. グループごとの最近傍結合 (Inner Logic) ---
    common_keys = set(base_process[match_col]) & set(target_process[match_col])
    results = []

    # カラム名の衝突回避用サフィックス
    lsuffix = ""
    rsuffix = "_target"

    for key in common_keys:
        # IDによるフィルタリング
        sub_base = base_process[base_process[match_col] == key]
        sub_target = target_process[target_process[match_col] == key]
        
        if sub_base.empty or sub_target.empty:
            continue
            
        try:
            # ここでは条件に合うものだけを取得（Inner Join）
            joined = gpd.sjoin_nearest(
                sub_base,
                sub_target,
                how="inner", 
                max_distance=max_dist,
                distance_col=distance_col_name,
                lsuffix=lsuffix,
                rsuffix=rsuffix
            )
            results.append(joined)
        except Exception as e:
            print(f"Key {key} の処理中にエラーが発生しました: {e}")

    # --- 3. 結合結果の集約と重複排除 (m:1の保証) ---
    if results:
        matched_df = pd.concat(results, ignore_index=True)
        
        # 距離が近い順、同じならデータの並び順でソート
        matched_df = matched_df.sort_values(by=[temp_id_col, distance_col_name])
        
        # ベース側のID(temp_id_col)ごとに最初の1件だけを残す
        matched_df = matched_df.drop_duplicates(subset=[temp_id_col], keep='first')
        
        # --- 4. ベースデータへの左結合 (Left Join) ---
        # ベースデータに含まれないカラム（ターゲット由来のカラム + 距離カラム）を特定
        base_cols = set(base_process.columns)
        matched_cols = set(matched_df.columns)
        # temp_id_colは結合キーとして必要なので除外しない
        new_cols = list((matched_cols - base_cols) | {temp_id_col})
        
        # 必要カラムのみを抽出したデータフレームを作成
        df_to_merge = matched_df[new_cols]
        
        # 一時IDを使って左結合
        final_gdf = pd.merge(
            base_process,
            df_to_merge,
            on=temp_id_col,
            how='left'
        )
    else:
        # マッチしたものが一つもない場合は、カラムだけ追加（全行NaN）するか、そのまま返す
        # ここでは距離カラムなどをNaNで追加して返す
        final_gdf = base_process.copy()
        final_gdf[distance_col_name] = float('nan')
        # ターゲット側のカラム構造が不明なため、最低限距離カラムのみ追加して返します
        # 必要であればここでgdf_targetのカラムをNaNで追加する処理を入れられます

    # --- 5. 後処理 ---
    # 一時IDの削除
    if temp_id_col in final_gdf.columns:
        final_gdf = final_gdf.drop(columns=[temp_id_col])
    
    # マージ操作でGeoDataFrameの属性が失われる場合があるため再設定
    if not isinstance(final_gdf, gpd.GeoDataFrame):
        final_gdf = gpd.GeoDataFrame(final_gdf, geometry=base_process.geometry.name, crs=base_process.crs)

    return final_gdf