#Auto MLOPS Suite


from pathlib import Path
import re
import subprocess

from core.llm import get_llm


SUPPORTED_PROVIDERS = ("aws", "gcp")


FENCED_BLOCK = re.compile(
    r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)",
    re.DOTALL,
)
TERRAFORM_LANGUAGES = ("terraform", "hcl", "tf")
TERRAFORM_BLOCK_START = re.compile(
    r"^\s*(?:terraform|provider|resource|variable|output|data|locals|module)\s*(?:\{|\")"
)


def extract_terraform_code(reply):
    """Extract the Terraform code from an LLM reply.

    If the reply has fenced code blocks, return the terraform/hcl one (or the
    first one). Otherwise drop any lines before the first line that starts a
    Terraform block, such as chat text like "Here is the configuration:".
    """
    text = reply.replace("\r\n", "\n").strip()

    blocks = [
        (language.lower(), body)
        for language, body in FENCED_BLOCK.findall(text)
    ]

    if blocks:
        for language, body in blocks:
            if language in TERRAFORM_LANGUAGES:
                return body.strip()

        return blocks[0][1].strip()

    lines = text.splitlines()

    for index, line in enumerate(lines):
        if TERRAFORM_BLOCK_START.match(line):
            return "\n".join(lines[index:]).strip()

    return text


def run_terraform(directory, command):
    """Run a Terraform command and return its result."""
    try:
        result = subprocess.run(
            ["terraform", f"-chdir={directory}", *command],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        return (
            result.returncode == 0,
            result.stdout + "\n" + result.stderr,
        )

    except FileNotFoundError:
        return False, "Terraform CLI is not installed."

    except subprocess.TimeoutExpired:
        return False, "Terraform command timed out."


def generate_terraform(
    provider,
    description,
    output_dir="generated",
    max_retries=3,
    on_attempt=None,
):
    """Generate and validate Terraform infrastructure code.

    on_attempt, if given, is called with the attempt number (1, 2, ...) at the
    start of each attempt, so callers such as the web page can report how many
    attempts were needed.
    """

    provider = provider.lower().strip()

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "Provider must be 'aws' or 'gcp'."
        )

    if not description.strip():
        raise ValueError(
            "Deployment description cannot be empty."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries cannot be negative."
        )

    output_path = Path(output_dir) / provider
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "main.tf"

    prompt = f"""
You are an experienced Terraform infrastructure engineer.

Generate a complete Terraform HCL configuration for {provider.upper()}.

Infrastructure requirements:
{description}

Rules:
- Return only valid Terraform HCL. No markdown, no code fences, no explanation.
- Include a terraform required_providers block.
- Use the appropriate AWS or Google provider.
- Declare every variable that is referenced.
- Use secure defaults.
- Do not hardcode credentials or secrets.
- Do not create unnecessary expensive infrastructure.
- Do not use deprecated resource arguments.
- Use separate resource blocks when required by the provider.
- Avoid duplicate Terraform blocks and resource names.
- Include appropriate outputs.
- Do not include explanations outside the HCL code.
"""

    llm = get_llm(temperature=0)

    last_error = ""

    for attempt in range(max_retries + 1):

        print(
            f"Generating Terraform configuration "
            f"(attempt {attempt + 1})..."
        )

        if on_attempt:
            on_attempt(attempt + 1)

        response = llm.invoke(prompt)

        terraform_code = response.content

        if not isinstance(terraform_code, str):
            raise ValueError(
                "LLM returned non-text content."
            )

        terraform_code = extract_terraform_code(
            terraform_code
        )

        if not terraform_code:
            raise ValueError(
                "LLM returned empty Terraform code."
            )

        file_path.write_text(
            terraform_code,
            encoding="utf-8",
        )

        # Format the generated Terraform file.
        fmt_success, fmt_output = run_terraform(
            output_path,
            ["fmt", "main.tf"],
        )

        if not fmt_success:
            last_error = fmt_output

        else:
            # Initialize Terraform without a remote backend.
            init_success, init_output = run_terraform(
                output_path,
                ["init", "-backend=false", "-input=false", "-no-color"],
            )

            if not init_success:
                last_error = init_output

            else:
                # Validate the generated configuration.
                valid, validation_output = run_terraform(
                    output_path,
                    ["validate", "-no-color"],
                )

                if valid:
                    print(
                        "Success! Terraform configuration is valid."
                    )

                    print(f"Saved to: {file_path}")

                    return file_path

                last_error = validation_output

        print("Terraform validation failed.")

        if "Terraform CLI is not installed" in last_error:
            raise RuntimeError(last_error)

        if attempt < max_retries:
            print("Asking AI to correct the errors...")

            prompt = f"""
You are an experienced Terraform engineer.

Correct the following Terraform configuration
for {provider.upper()}.

Original infrastructure requirements:
{description}

Current Terraform code:
{terraform_code}

Terraform errors:
{last_error}

Rules:
- Fix all reported errors.
- Preserve the original infrastructure requirements.
- Declare all referenced variables.
- Use current Terraform provider resource syntax.
- Return only valid Terraform HCL. No markdown, no code fences, no explanation.
"""

    raise RuntimeError(
        "Terraform generation failed after "
        f"{max_retries + 1} attempts.\n"
        f"Last error:\n{last_error}\n"
        f"Last generated file: {file_path}"
    )

