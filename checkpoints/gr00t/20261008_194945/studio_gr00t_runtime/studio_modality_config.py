from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
studio_config = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['camera_0', 'camera_1', 'camera_2', 'camera_3']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['joints']),
    'action': ModalityConfig(delta_indices=list(range(16)), modality_keys=['joints'], action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)]),
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),
}
register_modality_config(studio_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
