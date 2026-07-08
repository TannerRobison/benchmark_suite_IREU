import subprocess

result = subprocess.run(
    ["ngspice", "-b", "ng_test1.cir"],
    capture_output=True,
    text=True
)

print("Return code:", result.returncode)
print(result.stdout)

if result.stderr:
    print("Errors:")
    print(result.stderr)
