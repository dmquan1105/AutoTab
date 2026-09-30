# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core SheetBrain agent with iterative execution and validation."""

import os
import time
import zipfile
from typing import Any, Dict, Optional

from openai import OpenAI
from openpyxl import load_workbook
from PIL import Image

from ..config.settings import Config
from ..modules.execution import ExecutionModule
from ..modules.validation import ValidationModule
from ..utils.excel_toolkit import ExcelToolkit
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_WORKBOOK_FILE_SIZE = 100 * 1024 * 1024
MAX_WORKBOOK_UNCOMPRESSED_SIZE = 500 * 1024 * 1024
MAX_WORKBOOK_ARCHIVE_MEMBERS = 10000
MAX_WORKBOOK_COMPRESSION_RATIO = 100


class SheetBrain:
    """
    Excel analysis agent with iterative execution and validation.
    Supports iterative improvement through validation feedback.
    """

    @staticmethod
    def _public_execution_result(execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove generated code, tool output, and prompts from default diagnostics."""
        summary = execution_result.get('execution_summary', {})
        public_summary = {
            key: value for key, value in summary.items()
            if key not in {'execution_steps', 'final_answer'}
        }
        return {
            'success': execution_result.get('success', False),
            'answer': execution_result.get('answer', ''),
            'total_turns': execution_result.get('total_turns', 0),
            'execution_summary': public_summary,
        }

    @staticmethod
    def _public_validation_result(validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Return validation status without model-generated diagnostic text."""
        return {
            'validation_passed': validation_result.get('validation_passed', False),
            'confidence_score': validation_result.get('confidence_score', 0.0),
            'requires_reexecution': validation_result.get('requires_reexecution', False),
        }

    def __init__(self, excel_path: str, config: Optional[Config] = None):
        """
        Initialize the SheetBrain agent.

        Args:
            excel_path: Path to the Excel file
            config: Configuration object (if None, uses default config)
        """
        self.excel_path = excel_path
        self.config = config or Config()

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

        # Only reviewed helper functions are exposed to generated code.
        self.code_globals = {}
        self.code_locals = {}

        self._setup_excel_libraries()

        self.execution_module = ExecutionModule(
            self.client, self.config.deployment, self.code_globals, self.code_locals,
            self.config.extra_body,
        )
        self.validation_module = ValidationModule(
            self.client, self.config.deployment, self.config.extra_body,
        )

    def run(self, user_question: str, table_image: Optional[Image.Image] = None,
            max_turns: Optional[int] = None,
            enable_validation: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run the complete analysis with iterative improvement through validation.

        Args:
            user_question: The user's query about the Excel file
            table_image: Optional screenshot of relevant Excel sheet area
            max_turns: Maximum number of Execute-Validate iterations
            enable_validation: Whether to run the validation stage

        Returns:
            Dictionary containing complete analysis results
        """
        # Use config defaults if not specified
        max_turns = max_turns or self.config.max_turns
        enable_validation = enable_validation if enable_validation is not None else self.config.enable_validation

        logger.info("Starting iterative execution and validation")
        print("🚀 [SheetBrain] Starting iterative execution and validation...")
        print("="*80)

        overall_start_time = time.time()
        all_execution_results = []
        all_validation_results = []

        try:
            # ===== ITERATIVE EXECUTE-VALIDATE LOOP =====
            for iteration in range(max_turns):
                logger.info(f"Starting iteration {iteration + 1}/{max_turns}")
                print(f"\n🔄 [ITERATION {iteration + 1}/{max_turns}] EXECUTE-VALIDATE CYCLE")
                print("="*60)

                # ===== STAGE 2: EXECUTION =====
                print(f"💻 [ITERATION {iteration + 1}] EXECUTION MODULE")
                print("-" * 40)
                execution_start_time = time.time()

                execution_result = self.execution_module.run(user_question)
                execution_duration = time.time() - execution_start_time
                all_execution_results.append(execution_result)

                status_emoji = "✅" if execution_result["success"] else "❌"
                print(f"{status_emoji} [ITERATION {iteration + 1}] Execution completed in {execution_duration:.2f}s")
                print(f"🔄 [ITERATION {iteration + 1}] Total turns: {execution_result['total_turns']}")
                print(f"📊 [ITERATION {iteration + 1}] Code executions: {execution_result.get('execution_summary', {}).get('total_code_executions', 0)}")

                # ===== STAGE 3: VALIDATION (if enabled) =====
                if enable_validation:
                    logger.info(f"Running validation module for iteration {iteration + 1}")
                    print(f"\n🔍 [ITERATION {iteration + 1}] VALIDATION MODULE")
                    print("-" * 40)
                    validation_start_time = time.time()

                    validation_result = self.validation_module.reflect(execution_result, user_question)
                    validation_duration = time.time() - validation_start_time
                    all_validation_results.append(validation_result)

                    validation_emoji = "✅" if validation_result["validation_passed"] else "⚠️"
                    print(f"{validation_emoji} [ITERATION {iteration + 1}] Validation completed in {validation_duration:.2f}s")
                    print(f"🎯 [ITERATION {iteration + 1}] Confidence: {validation_result['confidence_score']:.2f}")
                    print(f"📋 [ITERATION {iteration + 1}] Validation: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}")

                    # Check if we should stop iterating
                    if validation_result['validation_passed']:
                        logger.info(f"Validation passed on iteration {iteration + 1}")
                        print(f"🎉 [SUCCESS] Validation passed on iteration {iteration + 1}!")
                        final_answer = validation_result.get('verified_answer', execution_result['answer'])
                        overall_success = True
                        confidence_score = validation_result['confidence_score']
                        validation_passed = True
                        break
                    elif not validation_result.get('requires_reexecution', True):
                        logger.warning("Validation indicates no further improvement possible")
                        print(f"🛑 [STOPPING] Validation indicates no further improvement possible")
                        final_answer = execution_result['answer']
                        overall_success = False
                        confidence_score = validation_result['confidence_score']
                        validation_passed = False
                        break
                    else:
                        logger.info(f"Issues found, preparing for iteration {iteration + 2}")
                        print(f"🔄 [CONTINUE] Issues found, preparing for iteration {iteration + 2}")
                        if iteration == max_turns - 1:
                            logger.warning("Reached maximum iterations without validation")
                            print(f"⚠️ [MAX ITERATIONS] Reached maximum iterations without validation")
                            final_answer = execution_result['answer']
                            overall_success = False
                            confidence_score = validation_result['confidence_score']
                            validation_passed = False
                else:
                    # No validation, use execution results directly
                    logger.info("Validation disabled, using execution results directly")
                    final_answer = execution_result['answer']
                    overall_success = execution_result['success']
                    confidence_score = 0.8 if execution_result['success'] else 0.3
                    validation_passed = execution_result['success']
                    break

            # Generate final report
            if all_validation_results:
                final_validation = all_validation_results[-1]
                issues_found = final_validation.get('issues_found', [])
                improvement_feedback = final_validation.get('improvement_feedback', '')
            else:
                issues_found = []
                improvement_feedback = ''

            # Detailed prompts are opt-in diagnostics.
            all_conversation_histories = []
            if self.config.include_diagnostics:
                for iteration_index, exec_result in enumerate(all_execution_results, 1):
                    conv_history = exec_result.get('conversation_history', [])
                    if conv_history:
                        all_conversation_histories.append({
                            'iteration': iteration_index,
                            'conversation_history': conv_history
                        })

            public_execution_results = all_execution_results
            public_validation_results = all_validation_results
            public_issues_found = issues_found
            public_improvement_feedback = improvement_feedback
            if not self.config.include_diagnostics:
                public_execution_results = [
                    self._public_execution_result(exec_result)
                    for exec_result in all_execution_results
                ]
                public_validation_results = [
                    self._public_validation_result(validation_result)
                    for validation_result in all_validation_results
                ]
                public_issues_found = []
                public_improvement_feedback = ''

            # ===== FINAL SUMMARY =====
            total_duration = time.time() - overall_start_time
            total_iterations = len(all_execution_results)

            logger.info(f"Analysis completed. Success: {overall_success}, Iterations: {total_iterations}")
            print("\n" + "="*80)
            print("🎯 [FINAL SUMMARY]")
            print("="*80)
            print(f"Overall Success: {'✅ YES' if overall_success else '❌ NO'}")
            print(f"Total Iterations: {total_iterations}")
            print(f"Final Answer: {final_answer}")
            print(f"Confidence Score: {confidence_score:.2f}/1.0")
            print(f"Validation Passed: {'✅ YES' if validation_passed else '❌ NO'}")
            print(f"Total Duration: {total_duration:.2f}s")
            print("="*80)

            return {
                "success": overall_success,
                "answer": final_answer,
                "confidence_score": confidence_score,
                "validation_passed": validation_passed,
                "total_iterations": total_iterations,
                "all_execution_results": public_execution_results,
                "all_validation_results": public_validation_results,
                "conversation_history": all_conversation_histories,
                "issues_found": public_issues_found,
                "improvement_feedback": public_improvement_feedback,
                "total_duration": total_duration,
                "user_question": user_question if self.config.include_diagnostics else ""
            }

        except Exception as e:
            error_duration = time.time() - overall_start_time
            logger.error(f"Critical error: {str(e)}")
            print(f"❌ [SheetBrain] Critical error: {str(e)}")
            print(f"⏱️ [SheetBrain] Failed after {error_duration:.2f}s")

            # Collect conversation histories even in error case when explicitly requested.
            all_conversation_histories = []
            if self.config.include_diagnostics:
                for iteration_index, exec_result in enumerate(all_execution_results, 1):
                    conv_history = exec_result.get('conversation_history', [])
                    if conv_history:
                        all_conversation_histories.append({
                            'iteration': iteration_index,
                            'conversation_history': conv_history
                        })

            public_execution_results = all_execution_results
            public_validation_results = all_validation_results
            if not self.config.include_diagnostics:
                public_execution_results = [
                    self._public_execution_result(exec_result)
                    for exec_result in all_execution_results
                ]
                public_validation_results = [
                    self._public_validation_result(validation_result)
                    for validation_result in all_validation_results
                ]

            return {
                "success": False,
                "answer": f"Analysis failed due to error: {str(e)}",
                "confidence_score": 0.0,
                "validation_passed": False,
                "total_iterations": len(all_execution_results),
                "all_execution_results": public_execution_results,
                "all_validation_results": public_validation_results,
                "conversation_history": all_conversation_histories,
                "issues_found": [f"Critical error: {str(e)}"],
                "improvement_feedback": "Review the error and try again",
                "total_duration": error_duration,
                "user_question": user_question if self.config.include_diagnostics else ""
            }

    def _setup_excel_libraries(self):
        """Setup Excel-related libraries and utilities."""
        try:
            self._validate_workbook_archive(self.excel_path)
            logger.info("Loading Excel file")
            start_time = time.time()
            workbook = load_workbook(
                self.excel_path,
                data_only=False,
                keep_links=False,
                keep_vba=False,
            )
            self.workbook = workbook
            load_time = time.time() - start_time
            logger.info(f"Excel file loaded in {load_time:.2f}s")
            print(f"📊 [Excel] Loaded in {load_time:.2f}s")

            # Create ExcelToolkit instance and add helper functions
            self.mcp_toolkit = ExcelToolkit(workbook, self.excel_path)
            excel_helpers = self.mcp_toolkit.get_helper_functions_dict()
            self.code_globals.update(excel_helpers)

            logger.info("Excel libraries loaded successfully")
            logger.info("Workbook contains %d sheets", len(workbook.sheetnames))
            print("📦 [SheetBrain] Excel libraries loaded successfully")
            print(f"📊 [SheetBrain] Workbook contains {len(workbook.sheetnames)} sheets")

        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            print(f"❌ [SheetBrain] Failed to import required libraries: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            print(f"❌ [SheetBrain] Failed to load Excel file: {e}")
            raise

    @staticmethod
    def _validate_workbook_archive(excel_path: str) -> None:
        """Reject oversized or suspicious Office archives before XML parsing."""
        if os.path.getsize(excel_path) > MAX_WORKBOOK_FILE_SIZE:
            raise ValueError("Workbook exceeds the compressed file-size limit")
        if not zipfile.is_zipfile(excel_path):
            raise ValueError("Workbook is not a valid Office ZIP archive")

        total_uncompressed = 0
        with zipfile.ZipFile(excel_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_WORKBOOK_ARCHIVE_MEMBERS:
                raise ValueError("Workbook contains too many archive members")
            for member in members:
                total_uncompressed += member.file_size
                if total_uncompressed > MAX_WORKBOOK_UNCOMPRESSED_SIZE:
                    raise ValueError("Workbook exceeds the uncompressed size limit")
                if member.flag_bits & 0x1:
                    raise ValueError("Encrypted workbook archive members are not allowed")
                if member.file_size > 0:
                    ratio = member.file_size / max(member.compress_size, 1)
                    if ratio > MAX_WORKBOOK_COMPRESSION_RATIO:
                        raise ValueError("Workbook archive has a suspicious compression ratio")
