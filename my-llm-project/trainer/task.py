from model import GroundedModel
from curriculum import GradePreK, Grade1, Grade2, Grade3, Grade4, Grade5, Grade6, Grade7, Grade8, Grade9, Grade10, Grade11, Grade12, ExtracurricularPhysics101, ExtracurricularCreativeWriting101, ExtracurricularMusicTheory101

def main():
    """Main function to run the training task."""
    print("Curriculum script initialized.")

    # 1. Initialize the model
    model = GroundedModel()
    print("\nModel initialized.")

    # 2. Define the curriculum as a sequence of grade classes
    curriculum_stages = [GradePreK, Grade1, Grade2, Grade3, Grade4, Grade5, Grade6, Grade7, Grade8, Grade9, Grade10, Grade11, Grade12]

    # 3. Sequentially process each grade in the curriculum
    for grade_class in curriculum_stages:
        grade_instance = grade_class()
        grade_name = grade_instance.__class__.__name__
        print(f"\n--- Starting {grade_name} Stage ---")
 
        dataset = grade_instance.generate_dataset(samples=5)
        print(f"Generated {len(dataset)} samples for {grade_name}:")
        for i, (prompt, answer) in enumerate(dataset):
            print(f"  Sample {i+1}: Prompt: {prompt} -> Answer: {answer}")

    print("\n--- Core Curriculum Complete ---")

    # 4. Process available extracurricular courses
    extracurricular_courses = [ExtracurricularPhysics101, ExtracurricularCreativeWriting101, ExtracurricularMusicTheory101]
    print("\nProcessing available extracurricular courses...")
    for course_class in extracurricular_courses:
        course_instance = course_class()
        course_name = course_instance.__class__.__name__
        print(f"\n--- Starting Extracurricular: {course_name} ---")
        dataset = course_instance.generate_dataset(samples=3)
        print(f"Generated {len(dataset)} samples for {course_name}:")
        for i, (prompt, answer) in enumerate(dataset):
            print(f"  Sample {i+1}: Prompt: {prompt} -> Answer: {answer}")

if __name__ == "__main__":
    main()
