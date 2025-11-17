from randomforest.pipelines.evaluator import run_evaluation, run_evaluation_gt
from randomforest.pipelines.inferencer import run_inference
from randomforest.pipelines.trainer import run_training
from randomforest.pipelines.extractor import run_extraction
from randomforest.config import parse_config           
from pcdet.utils import common_utils

# ======================================================================
# --- [수정] 메인 함수 (라우터) ---
# ======================================================================

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    if args.mode == 'extract':
        run_extraction(args, cfg, logger)
        
    elif args.mode == 'train':
        run_training(args, logger)
        
    elif args.mode == 'evaluate':
        run_evaluation(args, logger)
        
    elif args.mode == 'evaluate gt':
        run_evaluation_gt(args, cfg, logger)

    elif args.mode == 'inference': # [추가]
        run_inference(args, cfg, logger)
        
    else:
        logger.error(f"알 수 없는 모드입니다: {args.mode}")

if __name__ == '__main__':
    main()