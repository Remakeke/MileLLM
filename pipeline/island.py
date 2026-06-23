from config import TASK_TYPE, STAGNATION_THRESHOLD


class FeatureProgram:
    def __init__(self, feature_defs, feature_code, score, generation, island_id):
        self.feature_defs = feature_defs
        self.feature_code = feature_code
        self.score = score
        self.generation = generation
        self.island_id = island_id

    def is_better_than(self, other_score):
        if other_score is None:
            return True
        if TASK_TYPE == "classification":
            return self.score > other_score
        return self.score < other_score


class Island:
    def __init__(self, island_id, functional_identity, prompt_template):
        self.island_id = island_id
        self.functional_identity = functional_identity
        self.prompt_template = prompt_template
        self.best_program = None
        self.score_history = []
        self.stagnation_count = 0
        self._gen_has_improvement = False
        self.external_exemplars = []

    def update_best(self, program):
        if self.best_program is None or program.is_better_than(self.best_program.score):
            self.best_program = program
            self._gen_has_improvement = True
            self.score_history.append(program.score)
            return True
        self.score_history.append(self.best_program.score)
        return False

    def end_generation(self):
        if not self._gen_has_improvement:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self._gen_has_improvement = False

    def is_stagnant(self):
        return self.stagnation_count >= STAGNATION_THRESHOLD

    def add_external_exemplars(self, exemplars):
        self.external_exemplars.extend(exemplars)
        self.stagnation_count = 0

    def get_all_exemplars(self):
        exemplars = []
        if self.best_program is not None:
            exemplars.append(self.best_program)
        exemplars.extend(self.external_exemplars)
        return exemplars

    def clear_external_exemplars(self):
        self.external_exemplars = []
