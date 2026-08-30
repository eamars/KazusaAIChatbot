# Kazusa AI Chatbot

Kazusa 是一个与平台无关的角色大脑服务。适配器把平台事件规范化为
类型化消息信封；Brain 负责认知、对话、持久化和投递；RAG3 提供证据；
DSH 负责有界的多步骤任务执行。

## Plan 3：DSH 运行边界

生产任务路径只有 DSH 一条：

~~~text
适配器/调试客户端
  -> Brain 接收与认知
  -> 角色判断确有任务需要时发出 task_resolution_request
  -> 一个共享的 AgenticResolverRuntime
  -> dsh_task_binding.v1 写入 dsh_task_bindings
  -> 经过认证的 DSH Standard sidecar
  -> 原生工具与 14 个 kazusa_* 语义工具
  -> terminal、checkpoint、fault 或 canceled 结果
  -> 类型化 Brain observation
  -> 正常认知、对话、dispatcher 与适配器投递
~~~

语义阶段负责任务判断和工具选择。确定性代码负责契约校验、预算、
权限、持久化、幂等、限额、租约、CAS、authority、缓存失效和投递。
RAG 证据本身不能直接成为角色立场。

术语边界保持明确：capability 是系统允许的能力，affordance 是当前
上下文实际向模型展示的可用入口；episode 是一次受保护的运行记录；
fail closed 表示契约、权限或恢复条件不满足时不进入执行或投递。

直接后台准入返回临时、对模型隐藏的 TaskResolutionAdmissionV1，字段
严格只有 schema_version、accepted_task_id、background_work_job_id 和
task_session_id。它只是准入观察，不是 deferred 结果，不包含
authority，也不是 checkpoint 引用。只有 checkpoint 已提交后，才可
产生带有 DshResolutionRefV1 的
TaskResolutionResultV1(status="deferred")。

领取任务时创建或核验 dsh_task_binding.v1，并写入
dsh_task_bindings。operation_generation 与文档 revision CAS 为恢复、
继续和 terminal 重放提供围栏。worker 使用
task_orchestrator_worker_payload.v2，并只接受 open_dsh_resolution 或
continue_dsh_resolution。authority 在领取时重新签发；过期 authority、
租约丢失、sidecar 故障、格式错误和目录不匹配都 fail closed，只能从
持久化 binding 恢复。

## 已接受任务、就绪与排空

accepted_task_control.v1 是已接受任务的模型安全控制面，只允许
continue、summarize 和 cancel。状态查询为只读。控制始终作用于同一个
不透明 accepted-task/session binding，并经正常认知、对话、dispatcher
和适配器投递。队列租约、凭据、文件路径、原始 worker payload 和原始
证据不会进入模型上下文。

Brain 的就绪探针是经过认证的 GET /runtime/dsh/health，必须报告
configured、durable_store 和 cognition_judge。sidecar 的
system.health 只有在 route、Standard、semantic-worker、web、Brain、
catalog、policy、workspace、profile、release 与 store 全部匹配时才
能标记 ready。

只读排空审计命令如下：

~~~powershell
venv\Scripts\python scripts/check_dsh_plan3_drain.py --legacy-coding-workspace-root <abs-root> --format json
~~~

该命令统计五类受管控的历史遗留状态，不执行写入。部署和生产数据
变更仍然需要单独授权。

## DSH 语义工具目录

模型可见目录正好有 14 行。第 1 至第 13 行是 Plan 2 契约，保持
byte-identical；第 14 行是 Plan 3 唯一新增项。

1. kazusa_search_conversation_history
2. kazusa_read_conversation_entries
3. kazusa_summarize_conversation_participants
4. kazusa_search_memories
5. kazusa_read_memories
6. kazusa_remember_information
7. kazusa_revise_memory
8. kazusa_change_memory_lifecycle
9. kazusa_find_people_by_name
10. kazusa_read_person_profiles
11. kazusa_recall_active_context
12. kazusa_read_calendar_context
13. kazusa_inspect_attached_media
14. kazusa_inspect_public_media

kazusa_inspect_public_media 只接受一个 HTTP(S) 图片 URL 和视觉问题。
URL 不能含凭据或 fragment；DNS 解析结果不能落入 private、loopback、
link-local、multicast、reserved 或 unspecified 地址；每次重定向都要
重新检查，最多允许 3 次。请求超时为 15 秒，响应体上限为 6 MiB。
MIME 与 magic bytes 必须共同指向 PNG、JPEG、GIF 或 WebP；Pillow 必须
成功解码，宽高均须处于 1 至 8192。视觉证据的 source 是
dsh_public_media；原始字节和 base64 不进入模型契约。

新增行会产生新的 semantic catalog digest。没有未完成交互的 terminal
或 checkpointed V2 thread 会轮换到携带新 digest 的 segment；旧
authority 和 grant 立即 fail closed。切换前必须先排空仍开放的
pre-cutover 交互与 grant。

## 保留的责任边界

RAG3/local-context 仍负责普通聊天证据，包括 prewarm/cache owner、
会话与记忆、人物、日历和获准的 web 检索。Dialog 负责最终措辞；
consolidation、scheduler、reflection 与 future_speak 继续位于实时 DSH
任务会话之外。

旧后台和旧编码执行器的模型路由、旧 workspace、preflight 与 repair
设置均不是有效配置；完整的精确删除字段清单记录在当前 Plan 3 契约
中。当前任务路由使用六项 AGENTIC_RESOLVER_LLM_* 与 KAZUSA_DSH_*
sidecar/store/gateway 设置；RAG3 保留 planner、subagent 和 web
provider 路由族。

## 开发与验证

所有确定性测试使用项目虚拟环境：

~~~powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
~~~

live DB 测试需要隔离且可用的 MongoDB；live LLM 测试逐例执行，并检查
保存的输出与 trace。sidecar 的 build、typecheck 和 test 命令记录在
当前 Plan 3 证据台账中。

相关文档：

- docs/HOWTO.md
- docs/SUBAGENT_INTERFACES.md
- docs/architecture/agentic_resolver_architecture.md
- docs/architecture/cognition_contracts_design.md
- docs/architecture/dsh_integration_architecture.md
