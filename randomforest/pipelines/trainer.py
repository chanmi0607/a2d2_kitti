
import joblib

from randomforest.models.rf_trainer import run_training_stage1, run_training_stage2
import pandas as pd
from datetime import datetime
from pathlib import Path

def run_training(args, logger):
    """[수정] 'train' 모드 실행 (LabelEncoder 저장 추가)"""
    logger.info('----------------- Mode: Model Training -----------------')
    try:
        logger.info(f"학습 데이터 로드 중: {args.train_file}")
        df_train = pd.read_csv(args.train_file).fillna(0)
        logger.info(f" -> 총 {len(df_train)}개의 학습 샘플 발견.")
    except FileNotFoundError:
        logger.error(f"오류: 학습 파일을 찾을 수 없습니다. {args.train_file}"); return

    model_1 = run_training_stage1(df_train, logger)
    model_2, le_stage2 = run_training_stage2(df_train, logger) # ★ le 받기

    try:
        # 현재 시간 문자열 생성 (예: 20251118_133050)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(args.model1_out).parent
        backup_dir = base_dir / "backups" / f"train_{timestamp}"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"백업 폴더 생성됨: {backup_dir}")
        if model_1:
            joblib.dump(model_1, args.model1_out)
            logger.info(f"\n1단계 모델 저장 완료: {args.model1_out}")

            # (B) 백업
            backup_path_1 = backup_dir / Path(args.model1_out).name
            joblib.dump(model_1, backup_path_1)

        if model_2:
            joblib.dump(model_2, args.model2_out)
            logger.info(f"2단계 모델 저장 완료: {args.model2_out}")

            # (B) 백업
            backup_path_2 = backup_dir / Path(args.model2_out).name
            joblib.dump(model_2, backup_path_2)


        if le_stage2: # ★ le 저장
            joblib.dump(le_stage2, args.le_out)
            logger.info(f"LabelEncoder 저장 완료: {args.le_out}")

            # (B) 백업 경로
            backup_path_le = backup_dir / Path(args.le_out).name
            joblib.dump(le_stage2, backup_path_le)
            
    except Exception as e:
        logger.error(f"모델 또는 LabelEncoder 저장 중 오류 발생: {e}")
        
