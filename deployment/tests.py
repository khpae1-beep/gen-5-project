import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from deployment.generator import extract_terraform_code, generate_terraform

HCL = '''terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type = string
}'''


class ExtractTerraformCodeTests(SimpleTestCase):
    def test_fence_with_language_tag(self):
        reply = f'Here is the corrected Terraform configuration:\n\n```terraform\n{HCL}\n```\n\nThis creates a provider.'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_fence_with_hcl_tag_is_preferred_over_other_blocks(self):
        reply = f'Run this first:\n```bash\nterraform init\n```\nThen use:\n```hcl\n{HCL}\n```'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_fence_without_language_tag(self):
        reply = f'Sure!\n```\n{HCL}\n```'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_first_block_when_none_is_tagged_terraform(self):
        reply = f'```\n{HCL}\n```\nand an example:\n```\nresource "x" "y" {{}}\n```'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_unclosed_fence(self):
        self.assertEqual(extract_terraform_code(f'```terraform\n{HCL}\n'), HCL)

    def test_plain_hcl_is_unchanged(self):
        self.assertEqual(extract_terraform_code(HCL), HCL)
        self.assertEqual(extract_terraform_code(f'\n\n{HCL}\n\n'), HCL)

    def test_chat_text_before_the_code_is_dropped(self):
        reply = f'Here is the corrected Terraform configuration:\n\n{HCL}'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_each_block_keyword_can_start_the_code(self):
        starts = [
            'terraform {', 'provider "aws" {', 'resource "aws_s3_bucket" "b" {', 'variable "region" {',
            'output "id" {', 'data "aws_region" "current" {', 'locals {', 'module "vpc" {',
        ]
        for start in starts:
            with self.subTest(start):
                code = f'{start}\n}}'
                self.assertEqual(extract_terraform_code(f'I fixed the errors.\n{code}'), code)

    def test_chat_lines_starting_with_a_keyword_are_not_code(self):
        reply = f'data is stored in S3 with this configuration:\n{HCL}'
        self.assertEqual(extract_terraform_code(reply), HCL)

    def test_windows_line_endings(self):
        reply = f'Here it is:\r\n```terraform\r\n{HCL}\r\n```'.replace('\n', '\r\n').replace('\r\r\n', '\r\n')
        self.assertEqual(extract_terraform_code(reply), HCL)


class GenerateTerraformExtractionTests(SimpleTestCase):
    """The extraction is used for the first generation and every correction (no LLM or Terraform)."""

    def test_every_attempt_writes_only_the_hcl(self):
        replies = [
            f'Here is the Terraform configuration:\n```terraform\n{HCL}\n```',
            f'Here is the corrected Terraform configuration:\n\n```hcl\n{HCL}\n# fixed\n```',
        ]
        llm = mock.Mock()
        llm.invoke.side_effect = [SimpleNamespace(content=reply) for reply in replies]
        written = []

        def fake_terraform(directory, command):
            if command[0] == 'validate':
                written.append((Path(directory) / 'main.tf').read_text(encoding='utf-8'))
                return len(written) == 2, 'Error: something is wrong'
            return True, ''

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch('deployment.generator.get_llm', return_value=llm), \
                mock.patch('deployment.generator.run_terraform', side_effect=fake_terraform), \
                mock.patch('builtins.print'):
            path = generate_terraform('aws', 'An S3 bucket.', output_dir=tmp, max_retries=1)
            final = path.read_text(encoding='utf-8')

        self.assertEqual(written, [HCL, f'{HCL}\n# fixed'])
        self.assertEqual(final, f'{HCL}\n# fixed')
        prompts = [call.args[0] for call in llm.invoke.call_args_list]
        for prompt in prompts:
            self.assertIn('Return only valid Terraform HCL. No markdown, no code fences, no explanation.', prompt)
        # The correction prompt gets the extracted code, not the chat text around it.
        self.assertIn(HCL, prompts[1])
        self.assertNotIn('Here is the Terraform configuration', prompts[1])

    def test_on_attempt_reports_each_attempt_without_changing_the_output(self):
        def run(**kwargs):
            llm = mock.Mock()
            llm.invoke.side_effect = [SimpleNamespace(content=HCL)] * 3
            results = iter([(False, 'Error: one'), (False, 'Error: two'), (True, '')])

            def fake_terraform(directory, command):
                return next(results) if command[0] == 'validate' else (True, '')

            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch('deployment.generator.get_llm', return_value=llm), \
                    mock.patch('deployment.generator.run_terraform', side_effect=fake_terraform), \
                    mock.patch('builtins.print') as printed:
                generate_terraform('aws', 'An S3 bucket.', output_dir=tmp, **kwargs)
            return [call.args[0].replace(tmp, '<tmp>') for call in printed.call_args_list]

        attempts = []
        with_callback = run(on_attempt=attempts.append)
        self.assertEqual(attempts, [1, 2, 3])
        self.assertEqual(with_callback, run())
        self.assertIn('Generating Terraform configuration (attempt 3)...', with_callback)
