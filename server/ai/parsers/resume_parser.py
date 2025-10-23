# server/ai/parsers/resume_parser.py
"""
Resume/Portfolio PDF 파싱 및 청크 분할
"""
import pdfplumber
import io
from typing import List, Dict, Any, Optional
import re


class ResumeParser:
    """
    이력서/포트폴리오 PDF 파서

    PDF에서 텍스트를 추출하고 의미 단위로 청크를 분할합니다.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Args:
            chunk_size: 청크당 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수 (문맥 유지용)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_content: bytes) -> str:
        """
        PDF에서 텍스트 추출

        Args:
            pdf_content: PDF 파일 바이너리 내용

        Returns:
            str: 추출된 전체 텍스트

        Raises:
            Exception: PDF 파싱 실패 시
        """
        try:
            full_text = []

            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                print(f"[ResumeParser] PDF loaded: {len(pdf.pages)} pages")

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text:
                        cleaned_text = self._clean_text(text)
                        full_text.append(cleaned_text)
                        print(f"[ResumeParser]   Page {page_num}: {len(cleaned_text)} characters")

            result = "\n\n".join(full_text)
            print(f"[ResumeParser] Total extracted text: {len(result)} characters")

            return result

        except Exception as e:
            print(f"[ResumeParser] ✗ PDF parsing failed: {e}")
            raise Exception(f"Failed to parse resume PDF: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """텍스트 정리 - 불필요한 공백 제거"""
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text

    def split_into_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            metadata: 청크에 포함할 메타데이터

        Returns:
            List[Dict]: 청크 리스트
        """
        chunks = []
        text_length = len(text)

        if text_length == 0:
            print("[ResumeParser] Empty text, no chunks created")
            return chunks

        start = 0
        chunk_index = 0

        while start < text_length:
            end = start + self.chunk_size

            if end > text_length:
                end = text_length

            chunk_text = text[start:end]

            # 문장 경계에서 끊기
            if end < text_length:
                # 마지막 문장 종료 지점 찾기
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('!'),
                    chunk_text.rfind('?'),
                    chunk_text.rfind('。')  # 한국어 문장 부호
                )

                # 청크의 절반 이상에서 문장이 끝나면 그곳에서 자르기
                if last_sentence_end > self.chunk_size * 0.5:
                    end = start + last_sentence_end + 1
                    chunk_text = text[start:end]

            chunk = {
                "chunk_text": chunk_text.strip(),
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "metadata": metadata or {}
            }

            chunks.append(chunk)

            # 다음 청크 시작 위치 (오버랩 적용)
            start = end - self.chunk_overlap
            chunk_index += 1

        print(f"[ResumeParser] Created {len(chunks)} chunks")

        return chunks

    def parse_and_chunk(
        self,
        pdf_content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        PDF 파싱 + 청크 분할 (원스톱 처리)

        Args:
            pdf_content: PDF 파일 바이너리
            metadata: 청크 메타데이터

        Returns:
            Dict: {
                "full_text": str,
                "chunks": List[Dict],
                "total_chunks": int,
                "total_chars": int
            }
        """
        # 1. PDF → 텍스트
        full_text = self.parse_pdf(pdf_content)

        # 2. 텍스트 → 청크
        chunks = self.split_into_chunks(full_text, metadata)

        return {
            "full_text": full_text,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "total_chars": len(full_text)
        }
