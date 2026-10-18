import unittest
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

from app.api.agent import agent_chat
from app.schemas.platform import ChatRequest
from app.services.agent_service import classify_intent, route_agent, understand_query


class IntentRoutingTests(unittest.TestCase):
    def test_supported_courier_question_is_delivery_consultation(self):
        result = classify_intent("支持哪些快递")

        self.assertEqual(result["intentLabel"], "配送咨询")
        self.assertGreaterEqual(result["confidence"], 0.75)
        self.assertFalse(result["needConfirm"])
        self.assertFalse(result["transferHuman"])
        self.assertEqual(route_agent(result["intentLabel"], {}, "支持哪些快递")["targetAgent"], "BusinessConsultAgent")

    def test_identity_question_is_not_routed_to_human(self):
        result = classify_intent("你是谁")

        self.assertEqual(result["intentLabel"], "身份咨询")
        self.assertFalse(result["transferHuman"])
        self.assertEqual(route_agent(result["intentLabel"], {}, "你是谁")["targetAgent"], "GeneralQAAgent")

    def test_explicit_human_request_is_transferred(self):
        result = classify_intent("请帮我转人工")

        self.assertEqual(result["intentLabel"], "转人工")
        self.assertEqual(result["confidence"], 1.0)
        self.assertTrue(result["transferHuman"])

    def test_uncertain_model_human_label_does_not_trigger_transfer(self):
        embeddings = {
            "业务咨询": 0.0,
            "投诉": 0.0,
            "建议": 0.0,
            "转人工": 0.0,
            "闲聊": 0.0,
            "故障报修": 0.0,
            "订单咨询": 0.0,
            "配送咨询": 0.0,
            "身份咨询": 0.0,
        }
        with (
            patch(
                "app.services.agent_service._llm_intent",
                return_value=("转人工", 0.9, ""),
            ),
            patch(
                "app.services.agent_service._embedding_scores",
                return_value=embeddings,
            ),
        ):
            result = classify_intent("你是真人吗")

        self.assertNotEqual(result["intentLabel"], "转人工")
        self.assertFalse(result["transferHuman"])

    def test_asking_about_human_support_is_not_a_transfer_request(self):
        embeddings = {
            "业务咨询": 0.0,
            "投诉": 0.0,
            "建议": 0.0,
            "转人工": 0.0,
            "闲聊": 0.0,
            "故障报修": 0.0,
            "订单咨询": 0.0,
            "配送咨询": 0.0,
            "身份咨询": 0.0,
        }
        with (
            patch(
                "app.services.agent_service._llm_intent",
                return_value=("转人工", 0.9, ""),
            ),
            patch(
                "app.services.agent_service._embedding_scores",
                return_value=embeddings,
            ),
        ):
            result = classify_intent("请问人工客服工作时间")

        self.assertFalse(result["transferHuman"])
        self.assertNotEqual(result["intentLabel"], "转人工")

    def test_recent_context_is_passed_to_intent_classifier(self):
        context = [{"role": "assistant", "content": "您想了解配送方式吗？"}]
        embeddings = {
            "业务咨询": 0.0,
            "投诉": 0.0,
            "建议": 0.0,
            "转人工": 0.0,
            "闲聊": 0.0,
            "故障报修": 0.0,
            "订单咨询": 0.0,
            "配送咨询": 0.0,
            "身份咨询": 0.0,
        }
        with (
            patch(
                "app.services.agent_service._llm_intent",
                return_value=("配送咨询", 0.8, "配送方式"),
            ) as llm_intent,
            patch(
                "app.services.agent_service._embedding_scores",
                return_value=embeddings,
            ),
        ):
            classify_intent("有哪些", context)

        llm_intent.assert_called_once_with("有哪些", context)

    def test_rewrite_is_rejected_when_semantic_validation_fails(self):
        model_result = {
            "intentLabel": "订单咨询",
            "confidence": 0.9,
            "subIntent": "查询订单物流",
            "standaloneQuery": "查询我的订单当前物流状态",
            "queryType": "personal_order_status",
            "requiredSlots": ["orderNo"],
            "missingSlots": ["orderNo"],
            "entities": {},
        }
        order_intent = {
            "intentLabel": "订单咨询",
            "confidence": 0.9,
            "subIntent": "",
            "hitKeywords": [],
            "scoreDetail": {},
            "needConfirm": False,
            "transferHuman": False,
        }
        with (
            patch("app.services.agent_service._llm_understand_query", return_value=model_result),
            patch("app.services.agent_service._verify_query_rewrite", return_value=(False, "语义不一致")),
            patch("app.services.agent_service.classify_intent", return_value=order_intent),
        ):
            result = understand_query("有哪些配送方式")

        self.assertEqual(result["rewrittenQuery"], "有哪些配送方式")
        self.assertFalse(result["rewriteConsistent"])
        self.assertEqual(result["queryType"], "general_knowledge")

    def test_short_order_followup_is_recognized_as_live_status_query(self):
        order_intent = {
            "intentLabel": "订单咨询",
            "confidence": 0.5,
            "subIntent": "",
            "hitKeywords": [],
            "scoreDetail": {},
            "needConfirm": True,
            "transferHuman": False,
        }
        with (
            patch("app.services.agent_service._llm_understand_query", return_value=None),
            patch("app.services.agent_service.classify_intent", return_value=order_intent),
        ):
            result = understand_query(
                "我不记得订单号",
                [{"role": "assistant", "content": "您是想查询已有订单的物流状态吗？"}],
            )

        self.assertEqual(result["queryType"], "personal_order_status")
        self.assertEqual(result["intentInfo"]["intentLabel"], "订单咨询")
        self.assertIn("orderNo", result["missingSlots"])


class AgentChatFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_question_reaches_retrieval_and_answer_generation(self):
        reference_docs = [
            {
                "docId": "shipping#1",
                "content": "支持顺丰标准、顺丰次日达、京东物流和EMS。",
                "score": 0.9,
            }
        ]
        memory = {
            "workingMemory": [],
            "episodicMemory": [],
            "userProfile": {},
        }
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/agent/chat",
                "raw_path": b"/api/v1/agent/chat",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            }
        )
        body = ChatRequest(
            sessionId="session-delivery",
            userId="u-001",
            tenantId="default",
            query="支持哪些快递",
        )
        with (
            patch("app.api.agent.get_memory", new=AsyncMock(return_value=memory)),
            patch(
                "app.api.agent.understand_query",
                return_value={
                    "intentInfo": {
                        "intentLabel": "配送咨询",
                        "confidence": 0.95,
                        "subIntent": "",
                        "hitKeywords": [],
                        "scoreDetail": {},
                        "needConfirm": False,
                        "transferHuman": False,
                    },
                    "originalQuery": "支持哪些快递",
                    "rewrittenQuery": "支持的快递公司有哪些",
                    "rewriteConsistent": True,
                    "rewriteVerification": "same",
                    "queryType": "general_knowledge",
                    "entities": {},
                    "requiredSlots": [],
                    "missingSlots": [],
                },
            ),
            patch(
                "app.api.agent.retrieve_documents",
                return_value={
                    "rerankDocs": reference_docs,
                    "rawRetrieveList": reference_docs,
                    "rewriteQueries": ["支持哪些快递"],
                    "mrr": None,
                },
            ) as retrieve,
            patch(
                "app.api.agent.generate_answer",
                return_value="支持顺丰、京东物流和EMS。",
            ) as generate,
            patch("app.api.agent.enqueue_memory_save"),
            patch("app.api.agent.evaluate_sample", return_value={}),
        ):
            response = await agent_chat(body, request, {"sub": "u-001"})

        self.assertEqual(response.data["intentInfo"]["intentLabel"], "配送咨询")
        self.assertFalse(response.data["transferHuman"])
        self.assertEqual(response.data["agentName"], "BusinessConsultAgent")
        self.assertEqual(len(response.data["referenceDocs"]), 1)
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.args[0], "支持的快递公司有哪些")
        self.assertEqual(retrieve.call_args.args[4], ["支持哪些快递"])
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[0], "支持的快递公司有哪些")

    async def test_low_confidence_alone_does_not_skip_retrieval_or_transfer(self):
        memory = {
            "workingMemory": [{"role": "assistant", "content": "你好，有什么可以帮你？"}],
            "episodicMemory": [],
            "userProfile": {},
        }
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/agent/chat",
                "raw_path": b"/api/v1/agent/chat",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12346),
                "server": ("127.0.0.1", 8000),
            }
        )
        body = ChatRequest(
            sessionId="session-low-confidence",
            userId="u-001",
            tenantId="default",
            query="一些不确定的问题",
        )
        uncertain_intent = {
            "intentLabel": "业务咨询",
            "confidence": 0.2,
            "subIntent": "",
            "hitKeywords": [],
            "scoreDetail": {
                "llmFewShotScore": 0.2,
                "embeddingScore": 0.1,
                "regexKeywordScore": 0,
            },
            "needConfirm": False,
            "transferHuman": False,
        }
        understanding = {
            "intentInfo": uncertain_intent,
            "originalQuery": body.query,
            "rewrittenQuery": body.query,
            "rewriteConsistent": True,
            "rewriteVerification": "same",
            "queryType": "general_knowledge",
            "entities": {},
            "requiredSlots": [],
            "missingSlots": [],
        }
        with (
            patch("app.api.agent.get_memory", new=AsyncMock(return_value=memory)),
            patch(
                "app.api.agent.understand_query",
                return_value=understanding,
            ) as understand,
            patch(
                "app.api.agent.retrieve_documents",
                return_value={"rerankDocs": []},
            ) as retrieve,
            patch("app.api.agent.generate_answer", return_value="基于当前资料的回答。"),
            patch("app.api.agent.enqueue_memory_save"),
            patch("app.api.agent.evaluate_sample", return_value={}),
        ):
            response = await agent_chat(body, request, {"sub": "u-001"})

        self.assertFalse(response.data["transferHuman"])
        self.assertEqual(response.data["answer"], "基于当前资料的回答。")
        understand.assert_called_once()
        self.assertEqual(
            understand.call_args.args[1],
            memory["workingMemory"],
        )
        retrieve.assert_called_once()

    async def test_live_order_status_does_not_repeat_clarification(self):
        memory = {
            "workingMemory": [
                {"role": "assistant", "content": "您想查物流还是了解配送方式？"}
            ],
            "episodicMemory": [],
            "userProfile": {},
        }
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/agent/chat",
                "raw_path": b"/api/v1/agent/chat",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12347),
                "server": ("127.0.0.1", 8000),
            }
        )
        body = ChatRequest(
            sessionId="session-live-order",
            userId="u-001",
            tenantId="default",
            query="物流状态",
        )
        understanding = {
            "intentInfo": {
                "intentLabel": "订单咨询",
                "confidence": 0.4,
                "subIntent": "个人订单物流状态查询",
                "hitKeywords": [],
                "scoreDetail": {},
                "needConfirm": True,
                "transferHuman": False,
            },
            "originalQuery": body.query,
            "rewrittenQuery": body.query,
            "rewriteConsistent": True,
            "rewriteVerification": "same",
            "queryType": "personal_order_status",
            "entities": {},
            "requiredSlots": ["orderNo"],
            "missingSlots": ["orderNo"],
        }
        with (
            patch("app.api.agent.get_memory", new=AsyncMock(return_value=memory)),
            patch("app.api.agent.understand_query", return_value=understanding),
            patch("app.api.agent.retrieve_documents") as retrieve,
            patch("app.api.agent.generate_answer") as generate,
            patch("app.api.agent.enqueue_memory_save"),
            patch("app.api.agent.evaluate_sample", return_value={}),
        ):
            response = await agent_chat(body, request, {"sub": "u-001"})

        self.assertIn("无法直接连接订单或物流系统", response.data["answer"])
        self.assertFalse(response.data["transferHuman"])
        self.assertEqual(response.data["referenceDocs"], [])
        retrieve.assert_not_called()
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
