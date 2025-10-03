# src/cli/commands.py
import uvicorn
from src.train.train import train_detection_pipeline, train_classification_pipeline
from src.tracking.track import process_video_tracking
from src.api.main_api import app

def cmd_train_detect(args):
    return train_detection_pipeline(
        config_path=args.config,
        variant=args.variant,
        part1_weights=getattr(args, "part1_weights", None),
        part2_weights=getattr(args, "part2_weights", None),
        force_part1=getattr(args, "force_part1", False),
        force_part2=getattr(args, "force_part2", False),
    )

def cmd_train_classify(args):
    return train_classification_pipeline(
        config_path=args.config,
        variant=args.variant,
        base_weights=getattr(args, "weights", None),
    )

def cmd_track(args):
    out = process_video_tracking(
        video_path=args.video,
        det_weights=args.det_weights,
        cls_weights=args.cls_weights,
        config_path=args.config,
        out_dir=args.output,
    )
    print(out)
    return out

def cmd_api(args):
    app_import = "src.api.main_api:app"
    if args.reload:
        uvicorn.run(app_import, host=args.host, port=args.port, reload=True, workers=1)  # reload exige import string
    else:
        # sem reload pode passar o objeto diretamente OU a string
        from src.api.main_api import app
        uvicorn.run(app, host=args.host, port=args.port, reload=False, workers=1)