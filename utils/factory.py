def get_model(model_name, args):
    from models.ast_finetune import Learner
    return Learner(args)
