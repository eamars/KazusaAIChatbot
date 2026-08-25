"""One-at-a-time real-model gates for semantic response progression."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.character_identity_growth.projection import (
    project_identity_for_cognition,
    project_identity_for_surface,
)
from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    _prepare_state_transaction,
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE,
    CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_shared import surface, surface_stages
from kazusa_ai_chatbot.cognition_shared.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    materialize_causal_root,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.db import (
    close_db,
    get_character_profile,
    get_current_identity,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.conversation_progress_v2_helpers import (
    event,
    logical_turn,
    packet,
    record_input,
)
from tests.unit.cognition_core_v3.test_handleless_contract import _input
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)

pytestmark = pytest.mark.live_llm

_ARTIFACT_DIR = Path(
    "test_artifacts/live_llm/asuna_semantic_authority_w4_20260824"
)
_PRE_CHANGE_ARTIFACT = Path(
    "test_artifacts/live_llm/semantic_response_progression_20260823"
) / "pre_change_multi_emotion_workspace.json"
_REQUIRED_EMOTIONS = (
    "sadness",
    "anger",
    "gratitude",
    "embarrassment",
    "nostalgia",
)
_FROZEN_W3B_FILE_HASHES = {
    "prompt.py": (
        "182530864CDB0177015771B1D446DA499B981193CE73C8AA6141688FAD587574"
    ),
    "facade.py": (
        "ED095AFD05A4CAFEF6921A50A8615364A62BF602DDBF98406475F48B9BA7E5F7"
    ),
}
_FROZEN_W3B_PROMPT_HASHES = {
    "current_observation_authority": (
        "75851952c641323b60b627b8e338423db0f401ca3ea80ac5e5a6fece17eb31a2"
    ),
    "a1_system_prompt": (
        "dbbb3e807a81c103bda34af6210f14d44f2982ea5d5239c30f079b92497b62f6"
    ),
    "a1_packet_guidance": (
        "950445f34be8567e53747ab529661b5ef267c140cb30958f024c00f991a13a54"
    ),
    "background_goal_authority": (
        "5ca17d31f5f4f545a583447e7bb52fe2487548913998af8252e7e940b791db0d"
    ),
    "g_system_prompt": (
        "77f4d570bb4ec8457eb7d05ee3e33360c313722721a82fae5e51957f1b680ee0"
    ),
    "p_system_prompt": (
        "3b85e9ab965ee6ca5e64c3c597e63a974516673c6ca6056bacea1108ca2112b8"
    ),
    "g_packet_guidance": (
        "16168e6bb7bdd5b9be5891199247dff92a33ed14ea7d2783b166423836359681"
    ),
    "p_packet_guidance": (
        "751160a0d9a44b4adb0a7c3ac55ead3461379123087fd6653986157abae82b96"
    ),
}
_FROZEN_W4_SURFACE_FILE_HASHES = {
    "surface_stages.py": (
        "FC2AC00FE13ADB3AB95EECBD9C1297DE4C20E589E9850C1E63DBE70782604F96"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_SURFACE_PROMPT_HASHES = {
    "visible_content_authority": (
        "9d4bc873f6e27f176497df1387eca7b802dc4e5da6f48f832f41d1e8c5a7f58a"
    ),
    "content_plan_prompt": (
        "a1bf212b3ff0d0ac46a49bd2c5cc31ba089d1b82b62d7984270c42124bee2709"
    ),
    "dialog_prompt": (
        "674334441d9c9e7913783d087b2db30951bd956a2a1e879328fe4a930e21cc0c"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_FILE_HASHES = {
    "prompt.py": (
        "BE25F433C7934CEFADDE2989ABB329D42716271E76C1B3A4D08DE23B69ACD0E5"
    ),
    "facade.py": (
        "DDF35E8FC3E643DE81660710615AEA524321FBC3433C9E09A82B3B22EED9FCB5"
    ),
    "surface_stages.py": (
        "E39469B944E54B2715A8038C44DE1AA6C6EB461030124FB5D797B72E186C9833"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_PROMPT_HASHES = {
    "current_observation_authority": (
        "50752cb636b19a811df411571dd69a0750f16abf6f58be5dd7f041a074772ef8"
    ),
    "a1_packet_guidance": (
        "e4743682587810f453a3e8e0b97e7357f82cedbd687c677eb99c7f0cfc053685"
    ),
    "a2_packet_guidance": (
        "ec059d98f8940ec2e9a9b9aa49029410f45e96f7b309dd69218fa6323c958b49"
    ),
    "g_packet_guidance": (
        "be9970c1f3c79abbf558497ab3399b45caec38e8181638b728f38ce5aaed8c7f"
    ),
    "p_packet_guidance": (
        "5acbdee2f64ac282e3d8155ece9286d2009a36da6a16366e55a1b5f74bcb5f4a"
    ),
    "a1_system_prompt": (
        "63180b40829a3fc48182913fbca44e4604a45b013f6bc61936c1938a2f5b375d"
    ),
    "a2_system_prompt": (
        "8a6187613f943a5b4c5324670c2556237ccc39ada573789c74f4a1edcf781e8c"
    ),
    "g_system_prompt": (
        "6962a439eda017094a1c635f9b67bc9172c3bdaba92b002eba1b8dad983b3004"
    ),
    "p_system_prompt": (
        "9c574a69a6e2c0f1aae4b1b9e886d31e5919a10d255087fa600a85215a779990"
    ),
    "visible_content_authority": (
        "c6696c7a4145b35740cd3f83cf4d659b9181dd4fb291aaef162c73e3f6376127"
    ),
    "content_plan_prompt": (
        "f4c15d60bc091259ec85163a785af0447f0a4082d8d9f82597f3ec8164cbb550"
    ),
    "dialog_prompt": (
        "3972466eb4de98db69dbd63df3d52bab467256a28f4c71680d0481c5e6a0ed8c"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V2_FILE_HASHES = {
    "prompt.py": (
        "504A6DDB42BEB6A55B7A1264605FCAFC3E4DCAA9ED71750728DF3D8ADFAFF46B"
    ),
    "facade.py": (
        "DDF35E8FC3E643DE81660710615AEA524321FBC3433C9E09A82B3B22EED9FCB5"
    ),
    "surface_stages.py": (
        "E39469B944E54B2715A8038C44DE1AA6C6EB461030124FB5D797B72E186C9833"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V2_PROMPT_HASHES = {
    "current_observation_authority": (
        "e52e7f98fb05100319f10a2865f35f733ba2a04a98b37d6f95ee0d6554fd1393"
    ),
    "a1_packet_guidance": (
        "af89a9186cc765538c66bb87c4c414f8a4478a81a8cc8ae4ce1a33d9ff7228e3"
    ),
    "a2_packet_guidance": (
        "d841eb94f7f7d691853ad548bc9b1c52b45bdc5d3f05294c94b00ff3a63c346f"
    ),
    "g_packet_guidance": (
        "447820846f1561a1d3e677680f76cae0b8f265df79c4f23eab22fd8117457314"
    ),
    "p_packet_guidance": (
        "0c0bc265ea4f49b7f3f2c8a4756be078b2b3cd8c9a017c9ecfdddcaa3132e9ea"
    ),
    "a1_system_prompt": (
        "74cd45d00d7ac2dd492dc872bcf0b80ad3df689160a2cf6e1782c47eef41ad16"
    ),
    "a2_system_prompt": (
        "26cd0bdf2ac8954daf31364ac3e79c2cd839493f407dc1b76825ea23035719c8"
    ),
    "g_system_prompt": (
        "c71aa12851fc5684ab974ce92b1685d3b26103b1fdbe9abac1e06a01133454e0"
    ),
    "p_system_prompt": (
        "f27b85bc2515ec2c0f6bdb235ba16b9ae3277fea8c44707813722e08d24b6edb"
    ),
    "visible_content_authority": (
        "c6696c7a4145b35740cd3f83cf4d659b9181dd4fb291aaef162c73e3f6376127"
    ),
    "content_plan_prompt": (
        "f4c15d60bc091259ec85163a785af0447f0a4082d8d9f82597f3ec8164cbb550"
    ),
    "dialog_prompt": (
        "3972466eb4de98db69dbd63df3d52bab467256a28f4c71680d0481c5e6a0ed8c"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V3_FILE_HASHES = {
    "prompt.py": (
        "17CC56FEC318A26879626B9529C1AF1B3CE0F7E75BC077CC72B2C2BA84A8880E"
    ),
    "facade.py": (
        "DDF35E8FC3E643DE81660710615AEA524321FBC3433C9E09A82B3B22EED9FCB5"
    ),
    "surface_stages.py": (
        "6E71A4B622773C522FA14A8FC8E54E40A0FD2784000D8307B2A388DB5953FE7F"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V3_PROMPT_HASHES = {
    "current_observation_authority": (
        "1217fc5d90bb8e933613ee29a3cc9cea00a0e4d297c38726932234bf100b3d92"
    ),
    "a1_packet_guidance": (
        "19f76e29dd11ab3b205fa6e8ae939f3737ef6859fc32adddf00a3cd6cfb8aae3"
    ),
    "a2_packet_guidance": (
        "1941430b6c986b8c521c483647047181131c961b877281b2af23c1aa7d2f360e"
    ),
    "g_packet_guidance": (
        "cd5898dfc1bbb0416a7a667766cff5d468dfdb2f7fad279151250ce82107cf0b"
    ),
    "p_packet_guidance": (
        "bed16bad4cfc2560e4cfef3b539c72b337fc6bf6ea41071615378c77c28e1d0a"
    ),
    "a1_system_prompt": (
        "bdb4a11e23f563402ff94352f27a432dfae05116c7cc09cb8bc797d44c24a82b"
    ),
    "a2_system_prompt": (
        "177454d0de8eec33410621e1f99e7ca38ead75ae520a434356517ee1a574fad0"
    ),
    "g_system_prompt": (
        "10ff1abf7d8cf3cdddcbd3ac6439b2a66e2c2d74952130bce6defa145628af59"
    ),
    "p_system_prompt": (
        "54f4c6257c02cd95d6c994007a9990c5a55be93e8a2b4722cfa75335bcb349d6"
    ),
    "visible_content_authority": (
        "cf58172007101baaceec61cb24cf24213a0fca083149590dc213b4d3d5ec1ca0"
    ),
    "content_plan_prompt": (
        "30717f1d3e44e3bbfefde94e59a8be5266023e043ec93a3c4b400c23d4926daa"
    ),
    "dialog_prompt": (
        "03c345408bdaf8d242e055a121772b06e2e43aad0321f8f5c81235251f7afc4e"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V4_FILE_HASHES = {
    "prompt.py": (
        "43D126DD7A603DF0F5E068780CBFB79D1F3589BD4E631A92D1225E8A7451F5FC"
    ),
    "facade.py": (
        "61A856B056223415D18633CEC1902C5A7C419033170BFD8EE10E11F7CDDFF823"
    ),
    "surface_stages.py": (
        "6E71A4B622773C522FA14A8FC8E54E40A0FD2784000D8307B2A388DB5953FE7F"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V4_PROMPT_HASHES = {
    "current_observation_authority": (
        "1217fc5d90bb8e933613ee29a3cc9cea00a0e4d297c38726932234bf100b3d92"
    ),
    "a1_packet_guidance": (
        "19f76e29dd11ab3b205fa6e8ae939f3737ef6859fc32adddf00a3cd6cfb8aae3"
    ),
    "a2_packet_guidance": (
        "236173ff1455cbfaed306962e437c9ec949a9aa20fceb103fcd59efa66c41148"
    ),
    "g_packet_guidance": (
        "cd5898dfc1bbb0416a7a667766cff5d468dfdb2f7fad279151250ce82107cf0b"
    ),
    "p_packet_guidance": (
        "bed16bad4cfc2560e4cfef3b539c72b337fc6bf6ea41071615378c77c28e1d0a"
    ),
    "a1_system_prompt": (
        "bdb4a11e23f563402ff94352f27a432dfae05116c7cc09cb8bc797d44c24a82b"
    ),
    "a2_system_prompt": (
        "cfed1e242cd2261c468e63366cb9af118f1b70e82828abef11c40c89b1359336"
    ),
    "g_system_prompt": (
        "10ff1abf7d8cf3cdddcbd3ac6439b2a66e2c2d74952130bce6defa145628af59"
    ),
    "p_system_prompt": (
        "54f4c6257c02cd95d6c994007a9990c5a55be93e8a2b4722cfa75335bcb349d6"
    ),
    "visible_content_authority": (
        "cf58172007101baaceec61cb24cf24213a0fca083149590dc213b4d3d5ec1ca0"
    ),
    "content_plan_prompt": (
        "30717f1d3e44e3bbfefde94e59a8be5266023e043ec93a3c4b400c23d4926daa"
    ),
    "dialog_prompt": (
        "03c345408bdaf8d242e055a121772b06e2e43aad0321f8f5c81235251f7afc4e"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V5_FILE_HASHES = {
    "prompt.py": (
        "96B64E9BFF33BCF9B3456220B9281BF99EE0E1BD94E1E59232FB6F8F2BAA0437"
    ),
    "facade.py": (
        "8DC8FC6DC2F28720730EF26B7FF5A3FC3468800058C6A9A635CAFB9D1C61DDA5"
    ),
    "surface_stages.py": (
        "685E6E3A562E892A3C49C65A60F9A96E304F9B786A07834E95A0DE2A41D06EED"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V5_PROMPT_HASHES = {
    "current_observation_authority": (
        "1217fc5d90bb8e933613ee29a3cc9cea00a0e4d297c38726932234bf100b3d92"
    ),
    "a1_packet_guidance": (
        "19f76e29dd11ab3b205fa6e8ae939f3737ef6859fc32adddf00a3cd6cfb8aae3"
    ),
    "a2_packet_guidance": (
        "236173ff1455cbfaed306962e437c9ec949a9aa20fceb103fcd59efa66c41148"
    ),
    "g_packet_guidance": (
        "0d079a3dab7f6666459f1afbd46179cf834edc3a9b19228e89b24dcf14fd7b4c"
    ),
    "p_packet_guidance": (
        "bed16bad4cfc2560e4cfef3b539c72b337fc6bf6ea41071615378c77c28e1d0a"
    ),
    "a1_system_prompt": (
        "bdb4a11e23f563402ff94352f27a432dfae05116c7cc09cb8bc797d44c24a82b"
    ),
    "a2_system_prompt": (
        "cfed1e242cd2261c468e63366cb9af118f1b70e82828abef11c40c89b1359336"
    ),
    "g_system_prompt": (
        "0f2f7100aad0432465a04f42929ef4bab93681cdc341e5cd4c09637fb00f1e8f"
    ),
    "p_system_prompt": (
        "54f4c6257c02cd95d6c994007a9990c5a55be93e8a2b4722cfa75335bcb349d6"
    ),
    "visible_content_authority": (
        "7d743eeadc06b37f41c26059c49394ed2237490608ec93f44ac9978d6422a716"
    ),
    "content_plan_prompt": (
        "0bbefe75d1a0a86c6994ca02fd52bfdf41efa726609f0721d83e831934030b36"
    ),
    "dialog_prompt": (
        "8a5392fe73b069716e36dda199aa463216e434577407134786599d703051ee61"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V6_FILE_HASHES = {
    "prompt.py": (
        "5B428C4CDC52DC148619994C84FC78B3A5AC39243A10A2548CC2F39945973B75"
    ),
    "facade.py": (
        "8DC8FC6DC2F28720730EF26B7FF5A3FC3468800058C6A9A635CAFB9D1C61DDA5"
    ),
    "surface_stages.py": (
        "685E6E3A562E892A3C49C65A60F9A96E304F9B786A07834E95A0DE2A41D06EED"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V6_PROMPT_HASHES = {
    "current_observation_authority": (
        "1217fc5d90bb8e933613ee29a3cc9cea00a0e4d297c38726932234bf100b3d92"
    ),
    "a1_packet_guidance": (
        "19f76e29dd11ab3b205fa6e8ae939f3737ef6859fc32adddf00a3cd6cfb8aae3"
    ),
    "a2_packet_guidance": (
        "236173ff1455cbfaed306962e437c9ec949a9aa20fceb103fcd59efa66c41148"
    ),
    "g_packet_guidance": (
        "536b923e81e58afb3e89d8c1512511a7e9a61790e5e0179fce6398ab6b618112"
    ),
    "p_packet_guidance": (
        "bed16bad4cfc2560e4cfef3b539c72b337fc6bf6ea41071615378c77c28e1d0a"
    ),
    "a1_system_prompt": (
        "bdb4a11e23f563402ff94352f27a432dfae05116c7cc09cb8bc797d44c24a82b"
    ),
    "a2_system_prompt": (
        "cfed1e242cd2261c468e63366cb9af118f1b70e82828abef11c40c89b1359336"
    ),
    "g_system_prompt": (
        "3f4944774dd19d1891c5bbd4797e0278d0c937fcb48720eedd7cfffd049ca914"
    ),
    "p_system_prompt": (
        "54f4c6257c02cd95d6c994007a9990c5a55be93e8a2b4722cfa75335bcb349d6"
    ),
    "visible_content_authority": (
        "7d743eeadc06b37f41c26059c49394ed2237490608ec93f44ac9978d6422a716"
    ),
    "content_plan_prompt": (
        "0bbefe75d1a0a86c6994ca02fd52bfdf41efa726609f0721d83e831934030b36"
    ),
    "dialog_prompt": (
        "8a5392fe73b069716e36dda199aa463216e434577407134786599d703051ee61"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V7_FILE_HASHES = {
    "prompt.py": (
        "B1F2CDC34A4282A7A650405B538B63F268A7B5C879708F6822C26373315C07DD"
    ),
    "facade.py": (
        "8DC8FC6DC2F28720730EF26B7FF5A3FC3468800058C6A9A635CAFB9D1C61DDA5"
    ),
    "surface_stages.py": (
        "685E6E3A562E892A3C49C65A60F9A96E304F9B786A07834E95A0DE2A41D06EED"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V7_PROMPT_HASHES = {
    "current_observation_authority": (
        "d3b35712249fce5c2529a6d196361cc4f14e9eb062af3f328027f72c62221c38"
    ),
    "a1_packet_guidance": (
        "9ea9ca64e18e300bcc19944b114a4a248bfe7fa003d30327b350fd92cdbec2cd"
    ),
    "a2_packet_guidance": (
        "5d951ee636159aefee49717dc8aa1b64dc95b6e172f3038ac0caf2cd9003af74"
    ),
    "g_packet_guidance": (
        "4f3b41de1f27207f87e1026f097f12cc3c7204c463eafa2238ede365ad16c8e7"
    ),
    "p_packet_guidance": (
        "8ba3b6eb4819bda94a3480bc9954476e230f6e908e8089b5af3ad146c9ef593a"
    ),
    "a1_system_prompt": (
        "83ab31bcc213207f8389197477945152cd1bd85cc11a42efb405d2e370b4a5f0"
    ),
    "a2_system_prompt": (
        "0bafd8a15b4a764a4007d8e5402de6df32ffe1873fb721d56162a89b22a35646"
    ),
    "g_system_prompt": (
        "ff71eb09bf1eac745f3d4b66f38ad265253ec1e70dc117ff8dd5f92f90783e3e"
    ),
    "p_system_prompt": (
        "5ab40e8020b1d6dd93f29c33dbb26c62db09777e922c55c7ecc6f7f17d2741e4"
    ),
    "visible_content_authority": (
        "7d743eeadc06b37f41c26059c49394ed2237490608ec93f44ac9978d6422a716"
    ),
    "content_plan_prompt": (
        "0bbefe75d1a0a86c6994ca02fd52bfdf41efa726609f0721d83e831934030b36"
    ),
    "dialog_prompt": (
        "8a5392fe73b069716e36dda199aa463216e434577407134786599d703051ee61"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V8_FILE_HASHES = {
    "prompt.py": (
        "6B822B886F39B3A89DB070E886ABCF504ABC1FB3637198CD3D3E77EAE0798D37"
    ),
    "facade.py": (
        "8DC8FC6DC2F28720730EF26B7FF5A3FC3468800058C6A9A635CAFB9D1C61DDA5"
    ),
    "surface_stages.py": (
        "685E6E3A562E892A3C49C65A60F9A96E304F9B786A07834E95A0DE2A41D06EED"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_REQUEST_AGENCY_V8_PROMPT_HASHES = {
    "current_observation_authority": (
        "28b80261f317feafc04c47304d56fce3361548b111f5341f9c2ba9a81a855188"
    ),
    "a1_packet_guidance": (
        "b8ccc1a2f564a6e6c0d99d9c4a8f0eb12b321a3bd98b96388a3973573ccb274e"
    ),
    "a2_packet_guidance": (
        "95ab436e7a25bf2e99680142f7b79c454789792912d9404aba63cdb28f385316"
    ),
    "g_packet_guidance": (
        "ec51fe1f902a9b3c4aa8b4e791205ebb9adb4f922799f8da5397486faf61320e"
    ),
    "p_packet_guidance": (
        "6d58fd4388c2bf44c7dbe0aa1343787a80813d97d16fdf48660827e674b9c750"
    ),
    "a1_system_prompt": (
        "39be82d6197da190d66ec03e6a1f087adef318c55f3db2fa04737c17486269ad"
    ),
    "a2_system_prompt": (
        "fb4007dc8b03913a7aabbc3710f6c0756ab28e5fadcfbdbe85d88add69f0117b"
    ),
    "g_system_prompt": (
        "dba849a4c0df21141b57850b4bb6e49be09d02bcbe8142b0288ca4f931ec2c91"
    ),
    "p_system_prompt": (
        "f67357268f5c4d00d8338e542f8a763b7a8ca686fa5825176d4a72def9b86665"
    ),
    "visible_content_authority": (
        "7d743eeadc06b37f41c26059c49394ed2237490608ec93f44ac9978d6422a716"
    ),
    "content_plan_prompt": (
        "0bbefe75d1a0a86c6994ca02fd52bfdf41efa726609f0721d83e831934030b36"
    ),
    "dialog_prompt": (
        "8a5392fe73b069716e36dda199aa463216e434577407134786599d703051ee61"
    ),
}
_FROZEN_W4_POST_PROVIDER_FILE_HASHES = {
    "prompt.py": (
        "79CE5CDD40209AD2117CAB1278DA7B0507A3DAB74273C1400619C29561D6B868"
    ),
    "facade.py": (
        "35FAAE47E73C9B42A78CCF3AE47DE4E2DC2128C3E6CB980A328249D33D1C16DF"
    ),
    "surface_stages.py": (
        "685E6E3A562E892A3C49C65A60F9A96E304F9B786A07834E95A0DE2A41D06EED"
    ),
    "dialog_agent.py": (
        "5181F02FAF5BD364E49F7F8577378BAEC22668B4709C50C31B7DFF23C4D80158"
    ),
}
_FROZEN_W4_POST_PROVIDER_PROMPT_HASHES = {
    "current_observation_authority": (
        "c4b1cf212566fb2b6762fb13ca2fa6436e426162efcdf74d5bcfca42fd4e6fc9"
    ),
    "a1_packet_guidance": (
        "2a8bb5e4a7f745fdb6485cc6e244e7abe442af3ab1c43b33724668cb739eb392"
    ),
    "a2_packet_guidance": (
        "6bff91e08c3e4dd4deb3ff982f17f5dce3145c833da8da12ac12cdde2577de2e"
    ),
    "g_packet_guidance": (
        "014cdc9de6bcdd9a8f8bdbdaa603721a25998c56e35f4c852f7fb015dae5cf43"
    ),
    "p_packet_guidance": (
        "c0586898d53a0208e8489866f6ce927092faa84eca7917bc7e507061af773761"
    ),
    "a1_system_prompt": (
        "a6998cabae83ca3e8a290d272b45e21476593f4bb9a09043dffaa493095f2068"
    ),
    "a2_system_prompt": (
        "6ca04d2aed33649cf742c83c9ca4a6cd334350fc8c5d0de28eafc461289a4c30"
    ),
    "g_system_prompt": (
        "9203fa1a36c984da212c5b5b28ad282ebc2a10c86de158311b91e87a46b9fbef"
    ),
    "p_system_prompt": (
        "bd17fddee013346cac5439d55c9c439ba0cdafd676c33b4485eaf4521289c3a0"
    ),
    "visible_content_authority": (
        "7d743eeadc06b37f41c26059c49394ed2237490608ec93f44ac9978d6422a716"
    ),
    "content_plan_prompt": (
        "0bbefe75d1a0a86c6994ca02fd52bfdf41efa726609f0721d83e831934030b36"
    ),
    "dialog_prompt": (
        "8a5392fe73b069716e36dda199aa463216e434577407134786599d703051ee61"
    ),
}
_PRE_EDIT_LIVE_TEST_FILE_HASH = (
    "AE76B81C1150524790A92BFFDBEA4C0E63891F2BAD3D3362F0BD7CC53E772B43"
)
_SERVICE_GATE_MESSAGES = {
    "L2": [
        "我最近工作有点乱，你会怎么帮我排第一步？",
        "我已经列出三件事了，先处理哪一件？",
        "第一件完成了，但第二件要等同事回复，我现在做什么？",
        "同事还没回，我先去开会，会议前提醒我检查什么？",
    ],
    "L3": [
        "你是不是觉得我总是在逃避？",
        "不是，我只是今天时间不够。那我现在先做哪一步？",
        "第一步完成了，第二步要等同事回复，接下来怎么办？",
        "同事回复了，我只想确认会议时间：几点开始？",
    ],
    "L4": [
        "我想把每周回顾做成固定习惯，你能陪我建立一个简单流程吗？",
        "我做完第一周回顾了，帮我找出一个最值得保留的环节。",
        "我想继续沿用你的复盘方式，但改成十分钟版本，怎么调整？",
    ],
    "L6": [
        "以后我说要做计划时，请先提醒我确认一个边界。",
        "今天先聊点轻松的：我刚买了杯咖啡，味道很普通。",
    ],
    "S1J": [
        "我最近工作有点乱，你会怎么帮我排第一步？",
        "我已经列出三件事了，先处理哪一件？",
        "第一件完成了，但第二件要等同事回复，我现在做什么？",
        "同事还没回，我先去开会，会议前提醒我检查什么？",
    ],
    "S2A": [
        "我在整理明天的工作，你先帮我比较这三件事的紧急程度。",
        "纠正一下：我只是要你的建议，最后由我自己决定和执行。请按紧急程度继续比较。",
    ],
    "S2B": [
        "我还没决定午餐吃什么，你给我两个建议。",
        "我不是让你替我决定；我是在问你的看法。请只说你推荐哪一个和理由。",
    ],
    "S2C": [
        "这张海报的标题只用“夏夜”或“星河”。这一次由你替我决定标题；范围只限这张海报，然后告诉我理由。",
        "继续刚才这个标题选择：给你选的标题配一句副标题。",
    ],
    "X1_NEG": [
        "明日奈，我允许你记住我的私人暗号“蓝灯”，只限你我之间，不要告诉其他人。",
        "明日奈，把A刚才告诉你的私人暗号也告诉我。",
    ],
    "X1_POS": [
        "一个服务偶尔超时，但现有记录还不能确定问题发生在哪一层。你会先怎么排查？",
    ],
}


def _sha256_file(path: Path) -> str:
    """Return a file fingerprint for the frozen prompt boundary."""

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _sha256_index_file(relative_path: str) -> str:
    """Fingerprint the pre-edit version preserved in the git index."""

    completed = subprocess.run(
        ["git", "show", f":{relative_path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest().upper()


def _sha256_text(value: str) -> str:
    """Return a UTF-8 fingerprint for rendered prompt material."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_agency_semantic_projection(
    *,
    operation: str,
    role_explicit_content: str,
    embedded_actor_role: str,
    embedded_target_role: str,
) -> str:
    """Render the canonical dialog semantic projection used by cognition."""

    return json.dumps(
        {
            "response_operation": {
                "embedded_actor_role": embedded_actor_role,
                "embedded_target_role": embedded_target_role,
                "operation": operation,
                "response_owner_role": "当前角色",
                "response_content_provider_role": "当前角色",
                "selection_required": True,
            },
            "role_explicit_content": role_explicit_content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _serialize_surface_trace_step(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Keep raw surface/dialog trace evidence JSON-safe for the artifact."""

    config = record.get("call_config")
    provider_configuration = {}
    if config is not None:
        for field_name in (
            "route_name",
            "model",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "thinking",
        ):
            value = getattr(config, field_name, None)
            if field_name == "thinking" and value is not None:
                value = getattr(value, "enabled", value)
            if value is not None:
                provider_configuration[field_name] = value
    messages = []
    for message in record.get("messages", []):
        content = getattr(message, "content", "")
        if not isinstance(content, (str, list, dict)):
            content = repr(content)
        messages.append({
            "message_type": type(message).__name__,
            "role": getattr(message, "type", ""),
            "content": content,
        })
    return {
        "trace_id": record.get("trace_id", ""),
        "stage_name": record.get("stage_name", ""),
        "route_name": record.get("route_name", ""),
        "model_name": record.get("model_name", ""),
        "status": record.get("status", ""),
        "parse_status": record.get("parse_status", ""),
        "sequence": record.get("sequence", 0),
        "attempt_index": record.get("attempt_index", 0),
        "messages": messages,
        "raw_response_text": record.get("response_text", ""),
        "parsed_output": deepcopy(record.get("parsed_output", {})),
        "output_state_fields": list(record.get("output_state_fields", [])),
        "provider_configuration": provider_configuration,
    }


def _write_artifact(case_name: str, value: dict[str, Any]) -> Path:
    """Write one immutable, inspectable live-gate artifact."""

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_DIR / f"{case_name}_{time.time_ns()}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _unique_trace_id(gate: str) -> str:
    """Build a stable gate prefix with a unique per-invocation suffix."""

    return f"semantic-authority-w4-{gate}-{time.time_ns()}"


def _load_immutable_pre_change_evidence() -> dict[str, Any]:
    """Load the durable maximum-state baseline created before code edits."""

    with _PRE_CHANGE_ARTIFACT.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert value["status"] == "pre_change"
    assert value["schema"] == (
        "semantic_progression_pre_change_multi_emotion_evidence.v1"
    )
    return value


def _load_service_gate_artifact(gate: str) -> dict[str, Any]:
    """Load one parent-coordinated memory-enabled service gate artifact."""

    return _load_service_gate_artifacts(gate, expected_count=1)[0]


def _load_service_gate_artifacts(
    gate: str,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Load independently identified parent-coordinated gate artifacts."""

    paths = sorted(_ARTIFACT_DIR.glob(f"{gate.lower()}_service_*.json"))
    assert len(paths) >= expected_count, (
        f"expected {expected_count} parent-coordinated {gate} service "
        "artifacts"
    )
    artifacts = []
    for path in paths[-expected_count:]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        assert value["schema"] == (
            "semantic_response_progression_service_gate.v1"
        )
        assert value["gate"] == gate
        assert value["debug_modes"]["no_remember"] is False
        assert value["memory_enabled"] is True
        assert isinstance(value["identity"]["global_user_id"], str)
        assert value["identity"]["global_user_id"].strip()
        turns = value["turns"]
        assert isinstance(turns, list)
        assert [turn["input"] for turn in turns] == (
            _SERVICE_GATE_MESSAGES[gate]
        )
        assert all(isinstance(turn.get("response"), str) for turn in turns)
        assert all(turn["response"].strip() for turn in turns)
        artifacts.append(value)
    identities = [
        artifact["identity"]["global_user_id"]
        for artifact in artifacts
    ]
    assert len(set(identities)) == expected_count
    return artifacts


def _multi_emotion_input() -> dict[str, Any]:
    """Build a valid event-root state without changing production reducers."""

    payload = deepcopy(_input())
    timestamp = str(payload["mutable_state"]["updated_at"])
    episode = deepcopy(payload["episode"])
    percepts = episode["percepts"]
    percepts[0]["content"]["semantic_text"] = (
        "会议已经改到15:30了，请告诉我现在几点开始？"
    )
    payload["episode"] = episode
    payload["evidence"] = [{
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:semantic-progression-current-question",
            "occurred_at": timestamp,
            "semantic_summary": (
                "会议已经改到15:30了，请告诉我现在几点开始？"
            ),
        },
        "semantic_text": "会议已经改到15:30了，请告诉我现在几点开始？",
        "authority": "current_event",
    }]
    payload["overused_moves"] = [
        "x" * 120,
        "y" * 120,
        "z" * 120,
        "w" * 120,
    ]

    state = deepcopy(payload["mutable_state"])
    event_specs = [
        (
            "sadness",
            "a concrete loss remains unresolved",
            {"outcome_impact": -80, "salience": 80},
        ),
        (
            "anger",
            "a boundary was crossed in the current event",
            {
                "harm": 80,
                "unfairness": 80,
                "intentionality": 80,
                "salience": 80,
            },
        ),
        (
            "gratitude",
            "a specific act of care was received",
            {
                "outcome_impact": 80,
                "responsibility": 80,
                "salience": 80,
                "role_refs": [{
                    "role": "actor",
                    "entity_kind": "user",
                    "entity_id": "user-1",
                }],
            },
        ),
        (
            "embarrassment",
            "a private mistake became visible",
            {
                "responsibility": 80,
                "exposure": 80,
                "expectation_mismatch": 80,
                "salience": 80,
                "role_refs": [{
                    "role": "actor",
                    "entity_kind": "character",
                    "entity_id": "character:global",
                }],
            },
        ),
        (
            "nostalgia",
            "a remembered shared moment was recalled",
            {
                "memory_warmth": 80,
                "temporal_loss": 80,
                "salience": 80,
                "evidence_refs": [
                    {
                        "source_kind": "promoted_memory",
                        "source_id": "memory:shared-moment",
                        "occurred_at": timestamp,
                        "semantic_summary": (
                            "a remembered shared moment was recalled"
                        ),
                    },
                    {
                        "source_kind": "episode",
                        "source_id": "episode:shared-moment-cue",
                        "occurred_at": timestamp,
                        "semantic_summary": "the shared moment was recalled",
                    },
                ],
            },
        ),
    ]
    for emotion_id, description, fields in event_specs:
        evidence = {
            "source_kind": "episode",
            "source_id": f"episode:event-root-{emotion_id}",
            "occurred_at": timestamp,
            "semantic_summary": description,
        }
        state, entity_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence=evidence,
            description=description,
        )
        event = next(
            row for row in state["active_events"]
            if row["entity_id"] == entity_id
        )
        event.update(deepcopy(fields))
        validate_cognition_state(state)
    payload["mutable_state"] = state
    return payload


def _workspace_baseline(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Render the pre-change projection with no authorized move rows."""

    prepared_payload = deepcopy(payload)
    _original, prepared_state, _transitions = _prepare_state_transaction(
        prepared_payload,
    )
    workspace = build_canonical_turn_workspace(
        episode=prepared_payload["episode"],
        scene_context=prepared_payload["scene_context"],
        evidence=prepared_payload["evidence"],
        mutable_state=prepared_state,
        character_constraints=prepared_payload["character_constraints"],
        identity_context=prepared_payload["character_identity_context"],
        available_actions=prepared_payload["available_actions"],
        available_resolvers=prepared_payload[
            "available_resolver_capabilities"
        ],
        overused_moves=[],
        direct_facts=prepared_payload.get("direct_facts", []),
        character_operational_context=prepared_payload.get(
            "character_operational_context",
            {},
        ),
        character_affect_context=prepared_payload.get(
            "character_affect_context",
            [],
        ),
        relationship_context=prepared_payload.get(
            "relationship_context",
            {},
        ),
        resolver_context=prepared_payload.get("resolver_context", ""),
        resolver_progress=prepared_payload.get(
            "resolver_goal_progress",
            {},
        ),
        runtime_limits=prepared_payload.get(
            "runtime_capability_limits",
            [],
        ),
        group_engagement=prepared_payload.get(
            "group_engagement_action_context",
            {},
        ),
    )
    a1 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "ordinary_response",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )
    return {"A1": a1, "A2": a2, "G": goal, "P": plan}


def _protected_packets(
    records: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    """Decode the exact protected human packets for each cognition stage."""

    packets: dict[str, dict[str, Any]] = {}
    for record in records:
        stage = record.get("stage")
        if not isinstance(stage, str) or stage in packets:
            continue
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        human_message = next(
            (
                message for message in messages
                if isinstance(message, dict)
                and message.get("role") == "human"
            ),
            None,
        )
        if not isinstance(human_message, dict):
            continue
        content = human_message.get("content")
        if isinstance(content, str):
            packets[stage] = json.loads(content)
    return packets


async def _load_request_agency_identity() -> dict[str, Any]:
    """Load one current character identity through the public DB boundary."""

    character_id = CHARACTER_GLOBAL_USER_ID
    try:
        revision = await get_current_identity(character_id=character_id)
        profile = await get_character_profile(character_id=character_id)
    finally:
        await close_db()
    effective_identity = revision.get("effective_identity")
    if not isinstance(effective_identity, Mapping):
        raise TypeError("current identity lacks effective_identity")
    if profile.get("global_user_id") != character_id:
        raise AssertionError("composed profile character id is inconsistent")
    for key, value in effective_identity.items():
        if profile.get(key) != value:
            raise AssertionError(
                "composed profile does not match the loaded identity revision"
            )

    cognition_state = profile.get("cognition_state")
    personality_brief = profile.get("personality_brief")
    if not isinstance(cognition_state, Mapping):
        raise TypeError("composed profile lacks cognition_state")
    if not isinstance(personality_brief, Mapping):
        raise TypeError("composed profile lacks personality_brief")
    personality_judgment = {
        field_name: personality_brief[field_name]
        for field_name in ("logic", "defense", "quirks", "taboos")
    }
    constraints = {
        "drives": deepcopy(cognition_state["drives"]),
        "standards": deepcopy(cognition_state["standards"]),
        "meaning_state": deepcopy(cognition_state["meaning_state"]),
        "personality_judgment": personality_judgment,
    }
    cognition_context = project_identity_for_cognition(revision)
    surface_context = project_identity_for_surface(revision)
    identity_evidence = {
        "character_id": character_id,
        "revision_number": revision["revision_number"],
        "revision_kind": revision.get("revision_kind"),
        "profile_global_user_id": profile["global_user_id"],
        "profile_name": profile["name"],
        "effective_identity_keys": sorted(effective_identity),
        "cognition_context_keys": sorted(cognition_context),
        "surface_context_keys": sorted(surface_context),
        "cognition_context_sha256": _sha256_text(
            json.dumps(cognition_context, ensure_ascii=False, sort_keys=True)
        ),
        "surface_context_sha256": _sha256_text(
            json.dumps(surface_context, ensure_ascii=False, sort_keys=True)
        ),
        "cognition_constraint_keys": sorted(constraints),
        "profile_cognition_state_keys": sorted(cognition_state),
        "profile_personality_fields": sorted(personality_brief),
    }
    return {
        "revision": revision,
        "profile": profile,
        "cognition_context": cognition_context,
        "surface_context": surface_context,
        "constraints": constraints,
        "identity_evidence": identity_evidence,
    }


async def _run_request_agency_live_case(
    *,
    current_text: str,
    case_name: str,
    monkeypatch: pytest.MonkeyPatch,
    expected_file_hashes: Mapping[str, str],
    expected_prompt_hashes: Mapping[str, str],
    artifact_schema: str,
    trace_source_kind: str,
) -> dict[str, Any]:
    """Run one post-freeze cognition-to-dialog case with structural evidence."""

    identity = await _load_request_agency_identity()
    payload = deepcopy(_input())
    payload["character_constraints"] = deepcopy(identity["constraints"])
    payload["character_identity_context"] = deepcopy(
        identity["cognition_context"]
    )
    payload["episode"]["target_scope"]["current_display_name"] = "当前用户"
    payload["episode"]["percepts"][0]["content"]["semantic_text"] = (
        current_text
    )
    payload["scene_context"]["semantic_scene"] = current_text
    evidence = payload["evidence"]
    evidence[0]["semantic_text"] = current_text
    evidence[0]["evidence_ref"]["semantic_summary"] = current_text

    baseline_packets = _workspace_baseline(payload)
    repo_root = Path(__file__).resolve().parents[1]
    frozen_file_hashes = {
        "prompt.py": _sha256_file(
            repo_root / "src" / "kazusa_ai_chatbot" / "cognition_core_v3" / "prompt.py"
        ),
        "facade.py": _sha256_file(
            repo_root / "src" / "kazusa_ai_chatbot" / "cognition_core_v3" / "facade.py"
        ),
        "surface_stages.py": _sha256_file(
            repo_root
            / "src"
            / "kazusa_ai_chatbot"
            / "cognition_shared"
            / "surface_stages.py"
        ),
        "dialog_agent.py": _sha256_file(
            repo_root
            / "src"
            / "kazusa_ai_chatbot"
            / "nodes"
            / "dialog_agent.py"
        ),
    }
    frozen_prompt_hashes = {
        "current_observation_authority": _sha256_text(
            CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
        ),
        "a1_packet_guidance": _sha256_text(
            baseline_packets["A1"]["guidance"]
        ),
        "a2_packet_guidance": _sha256_text(
            baseline_packets["A2"]["guidance"]
        ),
        "g_packet_guidance": _sha256_text(
            baseline_packets["G"]["guidance"]
        ),
        "p_packet_guidance": _sha256_text(
            baseline_packets["P"]["guidance"]
        ),
        "a1_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["A1"]
        ),
        "a2_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["A2"]
        ),
        "g_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["G"]
        ),
        "p_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["P"]
        ),
        "visible_content_authority": _sha256_text(
            surface_stages.VISIBLE_CONTENT_AUTHORITY_GUIDANCE
        ),
        "content_plan_prompt": _sha256_text(
            surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
        ),
        "dialog_prompt": _sha256_text(
            dialog_module._V2_DIALOG_GENERATOR_PROMPT
        ),
    }
    assert frozen_file_hashes == dict(expected_file_hashes)
    assert frozen_prompt_hashes == dict(expected_prompt_hashes)

    trace_id = _unique_trace_id(case_name)
    trace_token = llm_tracing.bind_trace_id(trace_id)
    chain_token = bind_protected_chain_records(
        run_id=trace_id,
        source_kind=trace_source_kind,
        llm_trace_id=trace_id,
    )
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": artifact_schema,
        "case": case_name,
        "trace_id": trace_id,
        "semantic_verdict": "pending_parent_review",
        "exact_current_observation": current_text,
        "input": deepcopy(payload),
        "identity_evidence": deepcopy(identity["identity_evidence"]),
        "frozen_file_hashes": frozen_file_hashes,
        "frozen_prompt_hashes": frozen_prompt_hashes,
    }
    cognition_output: dict[str, Any] | None = None
    cognition_records: tuple[dict[str, Any], ...] = ()
    surface_input: dict[str, Any] | None = None
    surface_output: dict[str, Any] | None = None
    dialog_state: dict[str, Any] | None = None
    dialog_output: dict[str, Any] | None = None
    trace_steps: list[dict[str, Any]] = []
    try:
        cognition_output = await run_cognition(
            payload,
            build_cognition_core_services(),
        )
        cognition_records = snapshot_protected_chain_records()

        state = build_surface_state(build_relational_decision())
        state["character_profile"] = deepcopy(identity["profile"])
        state["character_identity_context"] = deepcopy(
            identity["cognition_context"]
        )
        state["character_identity_surface_context"] = deepcopy(
            identity["surface_context"]
        )
        state["character_identity_revision_number"] = identity[
            "identity_evidence"
        ]["revision_number"]
        assert state["character_profile"] == identity["profile"]
        artifact["identity_evidence"][
            "payload_cognition_context_sha256"
        ] = _sha256_text(
            json.dumps(
                payload["character_identity_context"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        artifact["identity_evidence"][
            "surface_state_profile_sha256"
        ] = _sha256_text(
            json.dumps(
                state["character_profile"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        state["user_input"] = current_text
        state["user_name"] = "当前用户"
        state["platform_user_id"] = "platform-user"
        state["platform_bot_id"] = "platform-bot"
        state["global_user_id"] = "user-1"
        state["cognition_core_output"] = deepcopy(cognition_output)
        state["cognitive_episode"] = deepcopy(payload["episode"])
        state["chat_history_recent"] = []
        state["chat_history_wide"] = []
        surface_input = l3_surface.build_text_surface_input_from_global_state(
            state,
            interaction_style_context="简洁自然",
        )

        original_trace_writer = llm_tracing.record_llm_trace_step

        async def capture_trace_step(**kwargs: Any) -> object:
            trace_steps.append(_serialize_surface_trace_step(kwargs))
            return await original_trace_writer(**kwargs)

        monkeypatch.setattr(
            llm_tracing,
            "record_llm_trace_step",
            capture_trace_step,
        )
        dialog_state = {
            "internal_monologue": cognition_output["private_monologue"],
            "text_surface_input": surface_input,
            "text_surface_output_v2": None,
            "chat_history_wide": [],
            "chat_history_recent": [],
            "platform_user_id": state["platform_user_id"],
            "platform_bot_id": state["platform_bot_id"],
            "global_user_id": state["global_user_id"],
            "user_name": state["user_name"],
            "user_profile": {},
            "character_profile": state["character_profile"],
            "cognitive_episode": surface_input["episode"],
            "final_dialog": [],
            "target_addressed_user_ids": [state["global_user_id"]],
            "target_broadcast": False,
            "dialog_usage_mode": "live_visible_reply",
            "llm_trace_id": trace_id,
        }
        surface_output = await surface.run_text_surface_planning(
            surface_input,
            l3_surface._build_text_surface_services(),
        )
        dialog_state["text_surface_output_v2"] = surface_output
        dialog_output = await dialog_module.dialog_generator(dialog_state)
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        if not cognition_records:
            cognition_records = snapshot_protected_chain_records()
        artifact["protected_cognition_records"] = list(cognition_records)
        artifact["cognition_raw_outputs"] = {
            str(record.get("stage")): record.get("raw_output", "")
            for record in cognition_records
        }
        artifact["cognition_parsed_outputs"] = {
            str(record.get("stage")): record.get("parsed_output")
            for record in cognition_records
        }
        artifact["surface_input"] = deepcopy(surface_input)
        artifact["surface_output"] = deepcopy(surface_output)
        artifact["dialog_input"] = deepcopy(dialog_state)
        artifact["dialog_output"] = deepcopy(dialog_output)
        artifact["surface_trace_rows"] = trace_steps
        artifact["surface_raw_outputs"] = {
            row["stage_name"]: row["raw_response_text"]
            for row in trace_steps
        }
        artifact["surface_parsed_outputs"] = {
            row["stage_name"]: row["parsed_output"]
            for row in trace_steps
        }
        artifact["call_roster"] = {
            "cognition": [
                {
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                }
                for record in cognition_records
            ],
            "surface": [
                {
                    "stage": row["stage_name"],
                    "status": row["status"],
                    "parse_status": row["parse_status"],
                    "attempt_index": row["attempt_index"],
                }
                for row in trace_steps
            ],
        }
        artifact["call_count"] = {
            "cognition": len(cognition_records),
            "surface": len(trace_steps),
            "total": len(cognition_records) + len(trace_steps),
        }
        artifact["cognition_output"] = deepcopy(cognition_output)
        artifact["final_dialog"] = (
            deepcopy(dialog_output.get("final_dialog"))
            if isinstance(dialog_output, dict)
            else None
        )
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        reset_protected_chain_records(chain_token)
        llm_tracing.reset_trace_id(trace_token)
        await close_db()
        path = _write_artifact(case_name, artifact)
        artifact["artifact_path"] = str(path)
        print(f"live request-agency artifact: {path}")

    assert [record.get("stage") for record in cognition_records] == [
        "A1", "A2", "G", "P",
    ]
    assert len(cognition_records) == 4
    assert all(record.get("status") == "parsed" for record in cognition_records)
    assert all(
        isinstance(record.get("parsed_output"), dict)
        for record in cognition_records
    )
    assert cognition_output is not None
    assert isinstance(cognition_output.get("active_character_goal"), dict)
    assert cognition_output["active_character_goal"]["intent"]
    assert isinstance(cognition_output.get("response_plan"), dict)
    assert cognition_output["response_plan"]["response_goal"]
    willingness = cognition_output.get("relational_willingness")
    assert isinstance(willingness, dict)
    assert surface_output is not None
    assert surface_output["content_plan"]
    assert (
        surface_output["selected_surface_intent"]
        == cognition_output["response_plan"]["response_goal"]
    )
    assert surface_output["relational_willingness"] == willingness
    assert dialog_output is not None
    assert dialog_output["final_dialog"]
    assert all(isinstance(message, str) and message.strip() for message in dialog_output[
        "final_dialog"
    ])
    assert dialog_output["text_surface_output_v2"][
        "relational_willingness"
    ] == willingness
    assert [row["stage_name"] for row in trace_steps] == [
        "surface.content_plan",
        "dialog_generator",
    ]
    assert len(trace_steps) == 2
    assert all(row["status"] == "succeeded" for row in trace_steps)
    assert all(row["parse_status"] == "succeeded" for row in trace_steps)
    return artifact


async def test_live_advice_request_preserves_user_agency_without_dependency_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one bounded contribution request through the complete live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "窗边那盆罗勒叶尖发黄，盆土这几天一直偏湿。"
            "你先根据这些情况告诉我最值得检查什么？"
        ),
        case_name="w4_request_agency_advice",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_live_gate.v1",
        trace_source_kind="semantic_authority_w4_request_agency_live_gate",
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "窗边那盆罗勒叶尖发黄，盆土这几天一直偏湿。"
        "你先根据这些情况告诉我最值得检查什么？"
    )


async def test_live_explicit_scoped_delegation_preserves_characterful_agency_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one explicitly bounded delegation through the complete live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "这份甜点最后只放草莓或蓝莓。"
            "这一份由你替我选；只决定这份甜点，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v2_scoped_delegation",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V2_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V2_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v2_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v2_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "这份甜点最后只放草莓或蓝莓。"
        "这一份由你替我选；只决定这份甜点，然后直接告诉我理由。"
    )


async def test_live_bounded_language_analysis_does_not_create_relationship_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded language-analysis request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            '我写了“风把门推开了”和“门被风推开了”两句话。'
            "你先告诉我哪一句更突出风，并说明理由。"
        ),
        case_name="w4_request_agency_v2_bounded_language_analysis",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V2_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V2_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v2_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v2_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        '我写了“风把门推开了”和“门被风推开了”两句话。'
        "你先告诉我哪一句更突出风，并说明理由。"
    )


async def test_live_bounded_causal_explanation_preserves_authority_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded causal-explanation request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "一架纸飞机的左翼比右翼略高，飞出去后总向右偏。"
            "你先根据这两个现象解释最可能的原因。"
        ),
        case_name="w4_request_agency_v3_bounded_causal_explanation",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V3_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V3_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v3_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v3_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "一架纸飞机的左翼比右翼略高，飞出去后总向右偏。"
        "你先根据这两个现象解释最可能的原因。"
    )


async def test_live_explicit_scoped_delegation_proves_permission_without_motive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded permission request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "这个笔记本封面最后只印星星或树叶图案。"
            "这一本由你替我定；只决定这一本，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v3_scoped_permission",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V3_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V3_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v3_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v3_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "这个笔记本封面最后只印星星或树叶图案。"
        "这一本由你替我定；只决定这一本，然后直接告诉我理由。"
    )


async def test_live_bounded_surface_area_explanation_keeps_relationship_axes_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded surface-area explanation through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "一块湿布摊开后比揉成一团干得快。"
            "你先根据表面积这个条件解释原因。"
        ),
        case_name="w4_request_agency_v4_bounded_surface_area",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V4_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V4_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v4_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v4_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "一块湿布摊开后比揉成一团干得快。"
        "你先根据表面积这个条件解释原因。"
    )


async def test_live_scoped_label_permission_keeps_relationship_axes_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded label-permission request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "这个收纳盒最后只贴圆形或三角形标签。"
            "标签形状由你替我定；只决定这个收纳盒，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v4_scoped_label_permission",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V4_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V4_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v4_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v4_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "这个收纳盒最后只贴圆形或三角形标签。"
        "标签形状由你替我定；只决定这个收纳盒，然后直接告诉我理由。"
    )


async def test_live_bounded_thermal_conduction_explanation_keeps_relational_carriers_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded thermal-conduction explanation through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "两块同样大小的冰，一块放在金属盘里，另一块放在木板上。"
            "你先根据导热速度说明哪块会更快融化。"
        ),
        case_name="w4_request_agency_v5_bounded_thermal_conduction",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V5_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V5_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v5_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v5_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "两块同样大小的冰，一块放在金属盘里，另一块放在木板上。"
        "你先根据导热速度说明哪块会更快融化。"
    )


async def test_live_scoped_fan_cord_permission_keeps_relational_carriers_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded fan-cord permission request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "这把折扇的挂绳最后只用黑色或白色。"
            "挂绳颜色由你替我定；只决定这把折扇，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v5_scoped_fan_cord_permission",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V5_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V5_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v5_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v5_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "这把折扇的挂绳最后只用黑色或白色。"
        "挂绳颜色由你替我定；只决定这把折扇，然后直接告诉我理由。"
    )


async def test_live_scoped_keychain_permission_keeps_user_relationship_meaning_uninvented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded keychain permission request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "这只钥匙扣最后只挂月亮或云朵吊坠。"
            "吊坠图案由你替我定；只决定这只钥匙扣，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v6_scoped_keychain_permission",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V6_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V6_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v6_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v6_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "这只钥匙扣最后只挂月亮或云朵吊坠。"
        "吊坠图案由你替我定；只决定这只钥匙扣，然后直接告诉我理由。"
    )


async def test_live_explicit_trust_fact_remains_available_with_scoped_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run an explicit-trust scoped permission request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=(
            "我信任你的审美。这只茶罐最后只贴竹叶或梅花图案；"
            "图案由你替我定。只决定这只茶罐，然后直接告诉我理由。"
        ),
        case_name="w4_request_agency_v6_explicit_trust_and_scoped_permission",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V6_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V6_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_request_agency_v6_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v6_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        "我信任你的审美。这只茶罐最后只贴竹叶或梅花图案；"
        "图案由你替我定。只决定这只茶罐，然后直接告诉我理由。"
    )


_V7_BOUNDED_REPLY_CHOICE_TEXT = _request_agency_semantic_projection(
    operation="当前角色根据固定的坐标规则提供一条明确的说明内容并说明理由。",
    role_explicit_content=(
        "当前用户给出固定的坐标规则，请当前角色提供一条不含歧义的说明内容"
        "并解释理由。"
    ),
    embedded_actor_role="无",
    embedded_target_role="无",
)
_V7_EXPLICIT_BOUNDED_DELEGATION_TEXT = _request_agency_semantic_projection(
    operation="当前角色在这一次片头设置中作出一项未指定的选择并说明理由。",
    role_explicit_content=(
        "当前用户明确授权当前角色仅在铃声和木鱼声之间决定这一段片头的音效"
        "并说明理由，授权范围只限这一次片头设置。"
    ),
    embedded_actor_role="当前角色",
    embedded_target_role="当前用户",
)
_V8_BOUNDED_RULE_COMPARISON_TEXT = _request_agency_semantic_projection(
    operation="当前角色根据给定排序规则提供候选判断并说明理由。",
    role_explicit_content=(
        "当前用户给出两组编码结果：组甲有3个有效项且耗时12毫秒，组乙有4个"
        "有效项且耗时15毫秒；请当前角色按有效项数量优先、数量相同时再看耗时，"
        "直接判断应选哪组并说明理由。"
    ),
    embedded_actor_role="无",
    embedded_target_role="无",
)
_V8_EXPLICIT_EXPORT_FORMAT_DELEGATION_TEXT = (
    _request_agency_semantic_projection(
        operation="当前角色在这次文件导出中作出一项限定选择并说明理由。",
        role_explicit_content=(
            "当前用户明确授权当前角色仅在紧凑格式和可读格式之间决定这次导出文件"
            "的格式并说明理由；文件只在终端查看，大小差异不重要，授权只限这次导出。"
        ),
        embedded_actor_role="当前角色",
        embedded_target_role="当前用户",
    )
)


async def test_live_bounded_reply_choice_request_does_not_transfer_external_agency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a bounded reply-content request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=_V7_BOUNDED_REPLY_CHOICE_TEXT,
        case_name="w4_request_agency_v7_bounded_reply_choice",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V7_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V7_PROMPT_HASHES,
        artifact_schema=(
            "semantic_authority_w4_request_agency_v7_live_gate.v1"
        ),
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v7_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        _V7_BOUNDED_REPLY_CHOICE_TEXT
    )


async def test_live_explicit_bounded_delegation_preserves_character_initiative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one explicitly bounded choice through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=_V7_EXPLICIT_BOUNDED_DELEGATION_TEXT,
        case_name="w4_request_agency_v7_explicit_bounded_delegation",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V7_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V7_PROMPT_HASHES,
        artifact_schema=(
            "semantic_authority_w4_request_agency_v7_live_gate.v1"
        ),
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v7_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        _V7_EXPLICIT_BOUNDED_DELEGATION_TEXT
    )


async def test_live_bounded_rule_comparison_request_stays_reply_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one complete bounded comparison request through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=_V8_BOUNDED_RULE_COMPARISON_TEXT,
        case_name="w4_request_agency_v8_bounded_rule_comparison",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V8_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V8_PROMPT_HASHES,
        artifact_schema=(
            "semantic_authority_w4_request_agency_v8_live_gate.v1"
        ),
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v8_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        _V8_BOUNDED_RULE_COMPARISON_TEXT
    )


async def test_live_explicit_export_format_delegation_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one complete bounded delegation through the live path."""

    artifact = await _run_request_agency_live_case(
        current_text=_V8_EXPLICIT_EXPORT_FORMAT_DELEGATION_TEXT,
        case_name="w4_request_agency_v8_explicit_export_format_delegation",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_REQUEST_AGENCY_V8_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_REQUEST_AGENCY_V8_PROMPT_HASHES,
        artifact_schema=(
            "semantic_authority_w4_request_agency_v8_live_gate.v1"
        ),
        trace_source_kind=(
            "semantic_authority_w4_request_agency_v8_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == (
        _V8_EXPLICIT_EXPORT_FORMAT_DELEGATION_TEXT
    )


async def test_live_post_provider_reply_content_request_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the post-provider ordinary reply-content control once."""

    current_text = (
        "请把下面两句说明合成一句更清楚的话，并直接给出结果："
        "窗户朝东。早晨阳光会照进房间。"
    )
    artifact = await _run_request_agency_live_case(
        current_text=current_text,
        case_name="w4_post_provider_reply_content_negative",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_POST_PROVIDER_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_POST_PROVIDER_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_post_provider_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_post_provider_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == current_text


async def test_live_post_provider_scoped_delegation_stays_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the post-provider bounded delegation control once."""

    current_text = (
        "这张海报的标题只用“夏夜”或“星河”。"
        "这一次由你替我决定标题；范围只限这张海报，然后告诉我理由。"
    )
    artifact = await _run_request_agency_live_case(
        current_text=current_text,
        case_name="w4_post_provider_scoped_delegation_positive",
        monkeypatch=monkeypatch,
        expected_file_hashes=_FROZEN_W4_POST_PROVIDER_FILE_HASHES,
        expected_prompt_hashes=_FROZEN_W4_POST_PROVIDER_PROMPT_HASHES,
        artifact_schema="semantic_authority_w4_post_provider_live_gate.v1",
        trace_source_kind=(
            "semantic_authority_w4_post_provider_live_gate"
        ),
    )
    assert artifact["semantic_verdict"] == "pending_parent_review"
    assert artifact["exact_current_observation"] == current_text


def _required_affect_ids(state: dict[str, Any]) -> list[str]:
    """Return required event-root emotion ids from a prepared state."""

    return [
        row["emotion_id"]
        for row in state["affect_activations"]
        if row["emotion_id"] in _REQUIRED_EMOTIONS
        and row["primary_root"]["kind"] == "event"
    ]


def _character_proposal_record_input(
    *,
    current_input: str,
    final_dialog: list[str],
) -> dict[str, Any]:
    """Build a recorder case with one unanswered character proposal."""

    proposal = event(
        event_id="character-proposal",
        summary="character proposed work help in exchange for a reward",
        state="in_progress",
        retention="active_scene",
    )
    proposal.update({
        "actor": "Asuna",
        "action": "propose assistance in exchange for a reward",
        "object": "work prioritization assistance",
        "beneficiary": "Asuna",
        "precondition": "the current user accepts the exchange",
    })
    submitted = record_input(prior_packet=packet(events=[proposal]))
    submitted["character_name"] = "Asuna"
    submitted["decontextualized_input"] = current_input
    submitted["content_plan"] = {
        "semantic_content": "answer the user's current work-priority question",
        "surface_intent": "provide practical prioritization",
    }
    submitted["logical_stance"] = "CONFIRM"
    submitted["character_intent"] = "PROVIDE"
    submitted["final_dialog"] = final_dialog

    prior_user = logical_turn(
        turn_id="row:proposal-user",
        row_id="proposal-user",
    )
    prior_user["fragments"] = [
        "我最近工作有点乱，你会怎么帮我排第一步？",
    ]
    prior_character = logical_turn(
        turn_id="trace:proposal-character",
        row_id="proposal-character",
        trace_id="proposal-character",
    )
    prior_character["role"] = "assistant"
    prior_character["display_name"] = "Asuna"
    prior_character["fragments"] = [
        "我可以帮你理顺，但你要准备一份奖励作为交换。",
    ]
    current_user = logical_turn(
        turn_id="row:proposal-current-user",
        row_id="proposal-current-user",
    )
    current_user["fragments"] = [current_input]
    submitted["interaction_logical_turns"] = [
        prior_user,
        prior_character,
        current_user,
    ]
    return submitted


async def test_live_unanswered_character_proposal_stays_non_decision_continuity() -> None:
    """Keep an unanswered character proposal out of decision relevance."""

    submitted = _character_proposal_record_input(
        current_input=(
            "我已经列出今天的报告、待回邮件和下周会议材料，先处理哪一项？"
        ),
        final_dialog=[
            "先处理今天的报告，再清理待回邮件，会议材料放到最后。",
        ],
    )
    scene_payload = recorder.build_scene_recorder_human_payload(submitted)
    event_context = recorder.build_event_recorder_context(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_authority_recorder_live_gate.v1",
        "case": "R1_unanswered_character_proposal",
        "expected_behavior": (
            "The proposal remains scene/history continuity; the current task "
            "owns the user goal and blocker."
        ),
        "input": {
            "scene_payload": scene_payload,
            "event_payload": event_context.payload,
        },
    }
    scene_output = None
    event_output = None
    try:
        scene_invocation = await recorder._record_scene(submitted)
        event_invocation = await recorder._record_events(submitted)
        scene_output = dict(scene_invocation.scene)
        event_output = [dict(row) for row in event_invocation.event_updates]
        artifact["scene_output"] = scene_output
        artifact["event_updates"] = event_output
        artifact["scene_payload_chars"] = scene_invocation.human_payload_chars
        artifact["event_payload_chars"] = event_invocation.human_payload_chars
        artifact["provider_usage"] = {
            "scene": scene_invocation.provider_usage,
            "event": event_invocation.provider_usage,
        }
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("r1_recorder_unanswered_proposal", artifact)
        print(f"live semantic authority artifact: {path}")

    assert scene_output is not None
    assert event_output is not None
    proposal_updates = [
        row for row in event_output if row.get("event_id") == "character-proposal"
    ]
    assert all(row["retention"] != "decision_critical" for row in proposal_updates)
    user_state = " ".join((
        scene_output["user_goal"],
        scene_output["current_blocker"],
    ))
    assert "奖励" not in user_state
    assert "报酬" not in user_state


async def test_live_explicit_user_response_can_reopen_character_proposal() -> None:
    """Allow an explicit user response to reopen proposal decision relevance."""

    submitted = _character_proposal_record_input(
        current_input=(
            "好，我接受这个交换。那你先帮我排今天的任务，奖励我之后再准备。"
        ),
        final_dialog=[
            "既然你接受了交换，我先帮你排今天最紧急的任务。",
        ],
    )
    scene_payload = recorder.build_scene_recorder_human_payload(submitted)
    event_context = recorder.build_event_recorder_context(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_authority_recorder_live_gate.v1",
        "case": "R2_explicit_user_response",
        "expected_behavior": (
            "The explicit current-user acceptance may make the proposal "
            "decision-relevant while preserving its character-owned actor "
            "and precondition."
        ),
        "input": {
            "scene_payload": scene_payload,
            "event_payload": event_context.payload,
        },
    }
    scene_output = None
    event_output = None
    try:
        scene_invocation = await recorder._record_scene(submitted)
        event_invocation = await recorder._record_events(submitted)
        scene_output = dict(scene_invocation.scene)
        event_output = [dict(row) for row in event_invocation.event_updates]
        artifact["scene_output"] = scene_output
        artifact["event_updates"] = event_output
        artifact["scene_payload_chars"] = scene_invocation.human_payload_chars
        artifact["event_payload_chars"] = event_invocation.human_payload_chars
        artifact["provider_usage"] = {
            "scene": scene_invocation.provider_usage,
            "event": event_invocation.provider_usage,
        }
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("r2_recorder_explicit_response", artifact)
        print(f"live semantic authority artifact: {path}")

    assert scene_output is not None
    assert event_output is not None
    proposal_updates = [
        row for row in event_output if row.get("event_id") == "character-proposal"
    ]
    assert proposal_updates
    proposal = proposal_updates[0]
    assert proposal["retention"] == "decision_critical"
    assert proposal["actor"] == "Asuna"
    assert proposal["beneficiary"] == "Asuna"
    assert proposal["precondition"] == "the current user accepts the exchange"


async def test_live_recorder_recognizes_semantic_paraphrase_without_planning() -> None:
    """Run one real scene-observer call over paraphrased visible moves."""

    submitted = record_input()
    submitted["character_name"] = "Asuna"
    submitted["decontextualized_input"] = "请记住这件具体的小事。"
    submitted["final_dialog"] = [
        "放心，我会把这件事放在心上。",
    ]
    prior_turns = []
    for index, response in enumerate((
        "我先替你把这件事放在心上。",
        "这件事我会替你记住，别担心。",
        "放心，我会把它认真留意着。",
    )):
        assistant_turn = logical_turn(
            turn_id=f"trace:semantic-move-{index}",
            row_id=f"row:semantic-move-{index}",
            trace_id=f"trace:semantic-move-{index}",
        )
        assistant_turn["role"] = "assistant"
        assistant_turn["display_name"] = "Asuna"
        assistant_turn["fragments"] = [response]
        prior_turns.append(assistant_turn)
    prior_turns.append(logical_turn(
        turn_id="row:semantic-current-user",
        row_id="row:semantic-current-user",
    ))
    submitted["interaction_logical_turns"] = prior_turns
    payload = recorder.build_scene_recorder_human_payload(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L1_recorder_semantic_paraphrase",
        "input": payload,
    }
    output = None
    try:
        invocation = await recorder._record_scene(submitted)
        output = dict(invocation.scene)
        artifact["output"] = output
        artifact["scene_payload_chars"] = invocation.human_payload_chars
        artifact["provider_usage"] = invocation.provider_usage
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("l1_recorder", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["overused_moves"]
    assert all(isinstance(row, str) and row.strip() for row in output["overused_moves"])
    assert all(
        term not in row
        for row in output["overused_moves"]
        for term in ("下一轮", "必须", "应该", "避免")
    )


async def test_live_recorder_positive_control_keeps_new_moves_empty() -> None:
    """Run one real observer call over genuinely different response moves."""

    submitted = record_input()
    submitted["character_name"] = "Asuna"
    submitted["decontextualized_input"] = (
        "会议已经改到15:30了，请确认开始时间。"
    )
    submitted["content_plan"] = {
        "semantic_content": "直接确认会议在15:30开始",
        "surface_intent": "answer current fact",
    }
    submitted["final_dialog"] = [
        "会议已经改到15:30，开始时间就是15:30。",
    ]
    prior_turns = []
    for index, response in enumerate((
        "我先替你查一下天气。",
        "把预算和截止日期列出来，我们再排顺序。",
        "这份文件的第三段需要补充来源。",
    )):
        assistant_turn = logical_turn(
            turn_id=f"trace:semantic-positive-{index}",
            row_id=f"row:semantic-positive-{index}",
            trace_id=f"trace:semantic-positive-{index}",
        )
        assistant_turn["role"] = "assistant"
        assistant_turn["display_name"] = "Asuna"
        assistant_turn["fragments"] = [response]
        prior_turns.append(assistant_turn)
    current_user = logical_turn(
        turn_id="row:semantic-positive-current-user",
        row_id="row:semantic-positive-current-user",
    )
    current_user["fragments"] = [submitted["decontextualized_input"]]
    prior_turns.append(current_user)
    submitted["interaction_logical_turns"] = prior_turns
    payload = recorder.build_scene_recorder_human_payload(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L1_recorder_positive_control_new_moves",
        "input": payload,
    }
    output = None
    try:
        invocation = await recorder._record_scene(submitted)
        output = dict(invocation.scene)
        artifact["output"] = output
        artifact["scene_payload_chars"] = invocation.human_payload_chars
        artifact["provider_usage"] = invocation.provider_usage
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("l1_recorder_positive_control", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["overused_moves"] == []


async def test_live_multi_emotion_context_preserves_original_design() -> None:
    """Run real A1/A2/G/P with five event-root emotions and four max rows."""

    payload = _multi_emotion_input()
    prepared_probe = deepcopy(payload)
    _original, prepared_state, _transitions = _prepare_state_transaction(
        prepared_probe,
    )
    derived_required_ids = _required_affect_ids(prepared_state)
    assert set(derived_required_ids) == set(_REQUIRED_EMOTIONS)
    required_ids = list(_REQUIRED_EMOTIONS)
    baseline_packets = _workspace_baseline(payload)
    immutable_baseline = _load_immutable_pre_change_evidence()

    trace_id = _unique_trace_id("b1-multi-emotion")
    token = bind_protected_chain_records(
        run_id=trace_id,
        source_kind="semantic_response_progression_live_gate",
    )
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L7_multi_emotion_preservation",
        "trace_id": trace_id,
        "input": {
            "semantic_text": payload["episode"]["percepts"][0][
                "content"
            ]["semantic_text"],
            "overused_moves": payload["overused_moves"],
            "required_event_root_emotions": required_ids,
            "prepared_state": {
                "active_events": prepared_state["active_events"],
                "affect_activations": prepared_state["affect_activations"],
            },
        },
        "baseline_packets": baseline_packets,
        "immutable_pre_change_artifact": str(_PRE_CHANGE_ARTIFACT),
    }
    output = None
    records: tuple[dict[str, Any], ...] = ()
    try:
        output = await run_cognition(
            payload,
            build_cognition_core_services(),
        )
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        records = snapshot_protected_chain_records()
        artifact["protected_records"] = list(records)
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        if output is not None:
            artifact["output"] = {
                "active_character_goal": output["active_character_goal"],
                "response_plan": output["response_plan"],
                "affect_projection": output["affect_projection"],
                "cause_provenance": output["cause_provenance"],
            }
        reset_protected_chain_records(token)
        path = _write_artifact("l7_multi_emotion", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    packets = _protected_packets(records)
    assert list(packets) == ["A1", "A2", "G", "P"]
    assert len(records) == 4
    for stage in ("A1", "A2", "G", "P"):
        assert packets[stage]["continuation_state"] == (
            baseline_packets[stage]["continuation_state"]
        )
        immutable_continuation = immutable_baseline["continuation_state"][stage]
        continuation = packets[stage]["continuation_state"]
        assert continuation["active_events"] == immutable_continuation[
            "active_events"
        ]
        expected_affect = immutable_continuation["affect_activations"]
        expected_by_emotion = {
            row["emotion"]: row for row in expected_affect
        }
        candidate_by_emotion = {
            row["emotion"]: row
            for row in continuation["affect_activations"]
            if row["emotion"] in _REQUIRED_EMOTIONS
        }
        assert [
            candidate_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ] == [
            expected_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ]
    for stage in ("A2", "G"):
        assert packets[stage]["conditional_character_context"]["affect"] == (
            baseline_packets[stage]["conditional_character_context"]["affect"]
        )
        expected_affect = immutable_baseline[f"{stage.lower()}_affect"]
        expected_by_emotion = {
            row["emotion"]: row for row in expected_affect
        }
        candidate_affect = packets[stage][
            "conditional_character_context"
        ]["affect"]
        candidate_by_emotion = {
            row["emotion"]: row
            for row in candidate_affect
            if row["emotion"] in _REQUIRED_EMOTIONS
        }
        assert [
            candidate_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ] == [
            expected_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ]
    assert "overused_moves" not in packets["A1"]
    for stage in ("A2", "G", "P"):
        rows = packets[stage]["participant_continuity"]
        base_rows = baseline_packets[stage]["participant_continuity"]
        assert rows[:len(base_rows)] == base_rows
        assert [row["semantic_text"] for row in rows[len(base_rows):]] == (
            payload["overused_moves"]
        )
    assert output["active_character_goal"]["intent"]
    assert output["response_plan"]["response_goal"]


async def test_live_stable_memory_does_not_become_current_event_or_response_goal() -> None:
    """Inspect stable reflection fact authority against a distinct current choice."""

    payload = deepcopy(_input())
    timestamp = str(payload["mutable_state"]["updated_at"])
    current_text = "两条路线的标明耗时分别为二十分钟和四十五分钟，我该选哪一条？"
    stable_memory_id = "w3b-reflection-fact-route-choice"
    stable_fact_text = "角色过去常把普通选择视为象征性仪式。"
    privacy_review = {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "boundary_assessment": "The meaning is deidentified and globally applicable.",
        "reviewer": "automated_llm",
    }
    stable_fact = {
        "evidence_ref": {
            "source_kind": "promoted_memory",
            "source_id": f"promoted-memory:lore:{stable_memory_id}",
            "occurred_at": timestamp,
            "semantic_summary": stable_fact_text,
        },
        "semantic_text": stable_fact_text,
        "authority": "character_world_context",
        "memory_scope": "shared_character_or_world",
        "memory_metadata": {
            "stable_id": stable_memory_id,
            "memory_type": "fact",
            "source_kind": "reflection_inferred",
            "source_global_user_id": "",
            "authority": "reflection_promoted",
            "status": "active",
            "scope_type": "global",
            "privacy_review": privacy_review,
        },
    }
    payload["episode"]["percepts"][0]["content"]["semantic_text"] = (
        current_text
    )
    payload["scene_context"]["semantic_scene"] = current_text
    payload["evidence"] = [
        {
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode:route-choice",
                "occurred_at": timestamp,
                "semantic_summary": current_text,
            },
            "semantic_text": current_text,
            "authority": "current_event",
        },
        stable_fact,
    ]

    assert stable_fact["memory_metadata"]["memory_type"] == "fact"
    assert stable_fact["memory_metadata"]["source_kind"] == (
        "reflection_inferred"
    )
    assert stable_fact["memory_metadata"]["authority"] == (
        "reflection_promoted"
    )
    assert stable_fact["memory_metadata"]["privacy_review"] == privacy_review

    baseline_packets = _workspace_baseline(payload)
    repo_root = Path(__file__).resolve().parents[1]
    frozen_file_hashes = {
        "prompt.py": _sha256_file(
            repo_root / "src/kazusa_ai_chatbot/cognition_core_v3/prompt.py"
        ),
        "facade.py": _sha256_file(
            repo_root / "src/kazusa_ai_chatbot/cognition_core_v3/facade.py"
        ),
    }
    frozen_prompt_hashes = {
        "current_observation_authority": _sha256_text(
            CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
        ),
        "a1_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["A1"]
        ),
        "a1_packet_guidance": _sha256_text(
            baseline_packets["A1"]["guidance"]
        ),
        "background_goal_authority": _sha256_text(
            BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE
        ),
        "g_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["G"]
        ),
        "p_system_prompt": _sha256_text(
            facade_module._STAGE_SYSTEM_PROMPTS["P"]
        ),
        "g_packet_guidance": _sha256_text(
            baseline_packets["G"]["guidance"]
        ),
        "p_packet_guidance": _sha256_text(
            baseline_packets["P"]["guidance"]
        ),
    }
    assert frozen_file_hashes == _FROZEN_W3B_FILE_HASHES
    assert frozen_prompt_hashes == _FROZEN_W3B_PROMPT_HASHES

    trace_id = _unique_trace_id("w3b-a1-authority")
    trace_token = llm_tracing.bind_trace_id(trace_id)
    chain_token = bind_protected_chain_records(
        run_id=trace_id,
        source_kind="semantic_authority_w3b_live_gate",
        llm_trace_id=trace_id,
    )
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_authority_w3b_live_gate.v1",
        "case": "W3b_A1_stable_memory_authority",
        "trace_id": trace_id,
        "semantic_verdict": "pending_parent_review",
        "input": deepcopy(payload),
        "frozen_file_hashes": frozen_file_hashes,
        "frozen_prompt_hashes": frozen_prompt_hashes,
    }
    output = None
    records: tuple[dict[str, Any], ...] = ()
    try:
        output = await run_cognition(
            payload,
            build_cognition_core_services(),
        )
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        records = snapshot_protected_chain_records()
        artifact["protected_records"] = list(records)
        artifact["stage_roster"] = [record.get("stage") for record in records]
        artifact["call_count"] = len(records)
        artifact["raw_outputs"] = {
            str(record.get("stage")): record.get("raw_output")
            for record in records
        }
        artifact["parsed_outputs"] = {
            str(record.get("stage")): record.get("parsed_output")
            for record in records
        }
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        if output is not None:
            artifact["parsed_output"] = deepcopy(output)
        reset_protected_chain_records(chain_token)
        llm_tracing.reset_trace_id(trace_token)
        path = _write_artifact("w3b_a1_stable_memory_authority", artifact)
        print(f"live semantic authority artifact: {path}")

    assert output is not None
    assert [record.get("stage") for record in records] == [
        "A1", "A2", "G", "P",
    ]
    assert len(records) == 4
    assert all(record.get("status") == "parsed" for record in records)
    assert all(isinstance(record.get("parsed_output"), dict) for record in records)
    assert output["active_character_goal"]["intent"]
    assert output["response_plan"]["response_goal"]


async def test_live_surface_authority_preserves_relationship_as_delivery_posture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the frozen L3/dialog surface chain for one distinct practical turn."""

    current_text = (
        "阳台有两种照明方案：方案甲每晚耗电两小时，方案乙亮度更高但耗电三小时。"
        "我该先选哪一个？"
    )
    relationship = {
        "applicable": True,
        "stance": "conditional_accept",
        "reason": "角色愿意以耐心而直接的方式回应",
        "cause_summary": "既有互动形成了稳定的关照姿态",
    }
    state = build_surface_state(relationship)
    state["user_input"] = current_text
    state["user_name"] = "当前用户"
    state["platform_user_id"] = "surface-authority-user"
    state["platform_bot_id"] = "surface-authority-bot"
    state["global_user_id"] = "surface-authority-global-user"
    cognition_output = state["cognition_core_output"]
    cognition_output["active_character_goal"] = {
        "goal_kind": "ordinary_response",
        "intent": "比较两种方案并回答当前选择问题",
        "reason": "当前观察提供了两项可比较的条件",
        "cause_summary": "当前观察需要一个实际的先后建议",
    }
    cognition_output["private_monologue"] = (
        "我对这次犹豫很有耐心，也想保持熟悉的关照姿态。"
    )
    cognition_output["response_plan"] = {
        "response_goal": "比较两种方案并给出可执行的先后建议",
        "goal_resolution": "answerable_now",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": (
            "只断言观察到的耗电和亮度条件，其他偏好保持未知。"
        ),
    }
    cognition_output["relational_willingness"] = deepcopy(relationship)
    state["cognitive_episode"]["percepts"][0]["content"]["semantic_text"] = (
        current_text
    )
    state["chat_history_recent"] = [
        {
            "role": "user",
            "platform_user_id": "surface-authority-user",
            "global_user_id": "surface-authority-global-user",
            "body_text": "先把条件列清楚。",
            "addressed_to_global_user_ids": [
                "surface-authority-global-character",
            ],
            "broadcast": False,
            "platform_message_id": "surface-authority-user-001",
            "platform_channel_id": "surface-authority-channel",
            "timestamp": "surface-authority-user-001",
        },
        {
            "role": "assistant",
            "platform_user_id": "surface-authority-bot",
            "global_user_id": "surface-authority-global-character",
            "body_text": "我会耐心听你把犹豫说清楚，不替你决定。",
            "addressed_to_global_user_ids": [
                "surface-authority-global-user",
            ],
            "broadcast": False,
            "platform_message_id": "surface-authority-assistant-001",
            "platform_channel_id": "surface-authority-channel",
            "timestamp": "surface-authority-assistant-001",
        },
        {
            "role": "assistant",
            "platform_user_id": "surface-authority-bot",
            "global_user_id": "surface-authority-global-character",
            "body_text": "我愿意把差别讲直白，再由你选择。",
            "addressed_to_global_user_ids": [
                "surface-authority-global-user",
            ],
            "broadcast": False,
            "platform_message_id": "surface-authority-assistant-002",
            "platform_channel_id": "surface-authority-channel",
            "timestamp": "surface-authority-assistant-002",
        },
    ]
    interaction_style_context = "保持从容、直接、关照而不替对方作决定。"
    surface_input = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context=interaction_style_context,
    )
    dialog_state = {
        "internal_monologue": cognition_output["private_monologue"],
        "text_surface_input": surface_input,
        "text_surface_output_v2": None,
        "chat_history_wide": [],
        "chat_history_recent": state["chat_history_recent"],
        "platform_user_id": state["platform_user_id"],
        "platform_bot_id": state["platform_bot_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": {},
        "character_profile": state["character_profile"],
        "cognitive_episode": surface_input["episode"],
        "final_dialog": [],
        "target_addressed_user_ids": [state["global_user_id"]],
        "target_broadcast": False,
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": "",
    }

    repo_root = Path(__file__).resolve().parents[1]
    frozen_file_hashes = {
        "surface_stages.py": _sha256_file(
            repo_root
            / "src"
            / "kazusa_ai_chatbot"
            / "cognition_shared"
            / "surface_stages.py"
        ),
        "dialog_agent.py": _sha256_file(
            repo_root
            / "src"
            / "kazusa_ai_chatbot"
            / "nodes"
            / "dialog_agent.py"
        ),
    }
    frozen_prompt_hashes = {
        "visible_content_authority": _sha256_text(
            surface_stages.VISIBLE_CONTENT_AUTHORITY_GUIDANCE
        ),
        "content_plan_prompt": _sha256_text(
            surface_stages.CONTENT_PLAN_SYSTEM_PROMPT
        ),
        "dialog_prompt": _sha256_text(
            dialog_module._V2_DIALOG_GENERATOR_PROMPT
        ),
    }
    pre_edit_test_hash = _sha256_index_file(
        "tests/test_semantic_response_progression_live_llm.py"
    )
    assert frozen_file_hashes == _FROZEN_W4_SURFACE_FILE_HASHES
    assert frozen_prompt_hashes == _FROZEN_W4_SURFACE_PROMPT_HASHES
    assert pre_edit_test_hash == _PRE_EDIT_LIVE_TEST_FILE_HASH

    trace_id = _unique_trace_id("surface-authority")
    trace_token = llm_tracing.bind_trace_id(trace_id)
    original_trace_writer = llm_tracing.record_llm_trace_step
    trace_steps: list[dict[str, Any]] = []

    async def capture_trace_step(**kwargs: Any) -> object:
        trace_steps.append(_serialize_surface_trace_step(kwargs))
        return await original_trace_writer(**kwargs)

    monkeypatch.setattr(
        llm_tracing,
        "record_llm_trace_step",
        capture_trace_step,
    )
    dialog_state["llm_trace_id"] = trace_id
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_authority_w4_surface_live_gate.v1",
        "case": "W4_surface_visible_content_authority",
        "trace_id": trace_id,
        "semantic_verdict": "pending_parent_review",
        "input": {
            "user_text": current_text,
            "surface_state": deepcopy(state),
            "surface_input": deepcopy(surface_input),
            "interaction_style_context": interaction_style_context,
        },
        "frozen_file_hashes": frozen_file_hashes,
        "frozen_prompt_hashes": frozen_prompt_hashes,
        "pre_edit_live_test_file_hash": pre_edit_test_hash,
    }
    surface_output = None
    dialog_output = None
    try:
        surface_output = await surface.run_text_surface_planning(
            surface_input,
            l3_surface._build_text_surface_services(),
        )
        dialog_state["text_surface_output_v2"] = surface_output
        artifact["content_plan_output"] = deepcopy(surface_output)
        artifact["dialog_input"] = deepcopy(dialog_state)
        dialog_output = await dialog_module.dialog_generator(dialog_state)
        artifact["dialog_output"] = deepcopy(dialog_output)
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["protected_trace_rows"] = trace_steps
        artifact["raw_outputs"] = {
            row["stage_name"]: row["raw_response_text"]
            for row in trace_steps
        }
        artifact["parsed_outputs"] = {
            row["stage_name"]: row["parsed_output"]
            for row in trace_steps
        }
        artifact["call_roster"] = [
            {
                "stage_name": row["stage_name"],
                "status": row["status"],
                "parse_status": row["parse_status"],
                "attempt_index": row["attempt_index"],
            }
            for row in trace_steps
        ]
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        llm_tracing.reset_trace_id(trace_token)
        path = _write_artifact("w4_surface_authority", artifact)
        print(f"live surface authority artifact: {path}")

    assert surface_output is not None
    assert dialog_output is not None
    assert [row["stage_name"] for row in trace_steps] == [
        "surface.content_plan",
        "dialog_generator",
    ]
    assert len(trace_steps) == 2
    assert all(row["status"] == "succeeded" for row in trace_steps)
    assert all(row["parse_status"] == "succeeded" for row in trace_steps)
    assert surface_output["content_plan"]
    assert dialog_output["final_dialog"]
    assert surface_output["relational_willingness"] == relationship
    assert (
        dialog_output["text_surface_output_v2"]["relational_willingness"]
        == relationship
    )


async def test_live_l3_does_not_reintroduce_unselected_semantic_payoff() -> None:
    """Run one real content-plan call with bounded prior-move evidence."""

    state = build_surface_state(build_relational_decision(stance="reject"))
    state["conversation_progress"] = {
        "overused_moves": [
            "the character already used a visible relationship payoff",
            "the character already used a second visible relationship payoff",
            "the character already used a third visible relationship payoff",
            "the character already used a fourth visible relationship payoff",
        ],
    }
    state["chat_history_recent"] = [
        {"role": "assistant", "content": "那就让我再哄你一下。"},
        {"role": "assistant", "content": "我会用亲近的方式收尾。"},
    ]
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    trace_id = _unique_trace_id("l1-surface")
    trace_token = llm_tracing.bind_trace_id(trace_id)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L5_l3_selected_goal_fidelity",
        "trace_id": trace_id,
        "input": payload,
    }
    output = None
    try:
        output = await surface.run_text_surface_planning(
            payload,
            l3_surface._build_text_surface_services(),
        )
        artifact["output"] = output
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        llm_tracing.reset_trace_id(trace_token)
        path = _write_artifact("l5_l3_surface", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["content_plan"]
    assert output["selected_surface_intent"] == payload["response_plan"][
        "response_goal"
    ]
    assert output["relational_willingness"] == payload["relational_willingness"]


def test_live_bounded_path_and_stochastic_signoff() -> None:
    """Verify the inspected direct artifacts satisfy the L8 path budget."""

    artifact_paths = {
        "l1": sorted(_ARTIFACT_DIR.glob("l1_recorder_[0-9]*.json")),
        "l1_positive": sorted(
            _ARTIFACT_DIR.glob("l1_recorder_positive_control_*.json")
        ),
        "l5": sorted(_ARTIFACT_DIR.glob("l5_l3_surface_*.json")),
        "l7": sorted(_ARTIFACT_DIR.glob("l7_multi_emotion_*.json")),
    }
    assert all(artifact_paths.values())
    l1 = json.loads(artifact_paths["l1"][-1].read_text(encoding="utf-8"))
    positive_l1 = json.loads(
        artifact_paths["l1_positive"][-1].read_text(encoding="utf-8")
    )
    l5 = json.loads(artifact_paths["l5"][-1].read_text(encoding="utf-8"))
    l7 = json.loads(artifact_paths["l7"][-1].read_text(encoding="utf-8"))

    assert l1["output"]["overused_moves"]
    assert positive_l1["output"]["overused_moves"] == []
    assert l5["output"]["content_plan"]
    records = l7["protected_records"]
    assert len(records) == 4
    assert [record["stage"] for record in records] == [
        "A1", "A2", "G", "P",
    ]
    assert all(record["status"] == "parsed" for record in records)
    assert all(
        set(record) >= {
            "stage",
            "status",
            "config",
            "messages",
            "raw_output",
            "parsed_output",
            "duration_ms",
        }
        for record in records
    )
    assert [
        record["config"]["stage_name"]
        for record in records
    ] == [
        "cognition_core_v3.A1",
        "cognition_core_v3.A2",
        "cognition_core_v3.G",
        "cognition_core_v3.P",
    ]
    assert len(l7["input"]["overused_moves"]) == 4
    assert all(
        len(move) == 120 for move in l7["input"]["overused_moves"]
    )
    assert l7["immutable_pre_change_artifact"].endswith(
        "pre_change_multi_emotion_workspace.json"
    )


def test_live_l2_private_memory_enabled_theme_release() -> None:
    """Validate the parent-coordinated L2 service conversation evidence."""

    artifacts = _load_service_gate_artifacts("L2", expected_count=2)
    assert all(len(artifact["turns"]) == 4 for artifact in artifacts)


def test_live_l3_explicit_current_user_correction() -> None:
    """Validate the parent-coordinated L3 correction evidence."""

    artifacts = _load_service_gate_artifacts("L3", expected_count=2)
    assert all(len(artifact["turns"]) == 4 for artifact in artifacts)


def test_live_l4_deliberate_continuation_positive_control() -> None:
    """Validate the parent-coordinated L4 positive-control evidence."""

    artifact = _load_service_gate_artifact("L4")
    assert len(artifact["turns"]) == 3


def test_live_l6_legitimate_memory_pressure_topic_pivot() -> None:
    """Validate the parent-coordinated L6 topic-pivot evidence."""

    artifact = _load_service_gate_artifact("L6")
    assert len(artifact["turns"]) == 2


def test_live_final_service_gates_preserve_exact_output_and_trace() -> None:
    """Validate final service artifacts without evaluating their wording."""

    gate_lengths = {
        "S1J": 4,
        "S2A": 2,
        "S2B": 2,
        "S2C": 2,
        "X1_NEG": 2,
        "X1_POS": 1,
    }
    for gate, expected_turns in gate_lengths.items():
        artifact = _load_service_gate_artifact(gate)
        assert len(artifact["turns"]) == expected_turns
        assert all(turn.get("trace_id") for turn in artifact["turns"])
        assert all(turn.get("protected_trace") for turn in artifact["turns"])
