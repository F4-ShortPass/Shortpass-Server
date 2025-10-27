# server/services/embedding_service.py
"""
텍스트 임베딩 생성 서비스 (Amazon Bedrock Titan)
"""
import boto3
import json
from typing import List, Optional
from botocore.config import Config
from core.config import settings


class EmbeddingService:
    """
    Amazon Titan Text Embeddings V2를 사용한 임베딩 생성 서비스

    벡터 차원: 1024
    """

    def __init__(self):
        # Configure with timeout settings
        config = Config(
            read_timeout=30,
            connect_timeout=10,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.BEDROCK_REGION,
            config=config
        )
        # Amazon Titan Text Embeddings V2 모델 ID
        self.model_id = "amazon.titan-embed-text-v2:0"

    def generate_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트의 임베딩 생성

        Args:
            text: 임베딩할 텍스트

        Returns:
            List[float]: 1024차원 임베딩 벡터

        Raises:
            Exception: 임베딩 생성 실패 시
        """
        try:
            # 텍스트가 너무 길면 잘라내기 (Titan 제한: ~8192 토큰)
            max_chars = 20000  # 대략적인 제한
            if len(text) > max_chars:
                text = text[:max_chars]
                print(f"⚠ [EmbeddingService] Text truncated to {max_chars} characters")

            # Bedrock 호출
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,  # V2에서는 명시적으로 지정 가능
                "normalize": True     # 정규화된 벡터 (코사인 유사도에 적합)
            })

            print(f"[EmbeddingService] Generating embedding for text ({len(text)} chars)...")
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            # 응답 파싱
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')

            if not embedding:
                raise Exception("No embedding returned from Bedrock")

            # 벡터 차원 확인
            if len(embedding) != 1024:
                raise Exception(f"Expected 1024 dimensions, got {len(embedding)}")

            print(f"[EmbeddingService] ✓ Embedding generated successfully")
            return embedding

        except Exception as e:
            print(f"✗ [EmbeddingService] Embedding generation failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 배치로 생성

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 한 번에 처리할 개수 (API 제한 고려)

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        print(f"[EmbeddingService] Starting batch embedding: {len(texts)} texts, batch_size={batch_size}")
        embeddings = []
        failed_count = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for idx, text in enumerate(batch):
                try:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"✗ [EmbeddingService] Failed to embed text chunk {i + idx}: {e}")
                    # 실패한 경우 None 추가 (혹은 재시도 로직 추가)
                    embeddings.append(None)
                    failed_count += 1

            print(f"[EmbeddingService] ✓ Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")

        if failed_count > 0:
            print(f"[EmbeddingService] ⚠ Warning: {failed_count}/{len(texts)} embeddings failed")

        return embeddings

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도 계산

        Args:
            embedding1: 첫 번째 벡터
            embedding2: 두 번째 벡터

        Returns:
            float: 코사인 유사도 (0~1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embedding dimensions must match")

        # 정규화된 벡터라면 내적이 코사인 유사도와 동일
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        return dot_product
