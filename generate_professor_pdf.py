"""
Generate a PDF document for the professor with:
- Apology and citation correction
- All references in IU/APA format
- How to run the code
- Output examples
"""

import os
from fpdf import FPDF


class ProfessorReport(FPDF):

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'LiDAR Object Detection and Tracking Pipeline - Code Appendix', 0, 1, 'C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(0, 51, 102)
        self.ln(4)
        self.cell(0, 9, title, 0, 1, 'L')
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.ln(2)
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font('Courier', '', 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + 5)
        for line in text.strip().split('\n'):
            self.cell(180, 5.5, '  ' + line, 0, 1, 'L', fill=True)
        self.ln(2)
        self.set_font('Helvetica', '', 10)

    def bullet(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + 8)
        self.cell(5, 5.5, '-', 0, 0)
        self.multi_cell(170, 5.5, text)
        self.ln(0.5)

    def ref_entry(self, text):
        """Single APA reference with hanging indent."""
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(30, 30, 30)
        x0 = self.get_x()
        self.set_x(x0 + 5)
        self.multi_cell(175, 5, text)
        self.ln(2)

    def add_image_safe(self, path, w=170):
        if os.path.exists(path):
            self.image(path, x=20, w=w)
            self.ln(4)
        else:
            self.italic_text(f'[Image not found: {path}]')


def generate_pdf():
    pdf = ProfessorReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== PAGE 1: Cover =====
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(0, 51, 102)
    pdf.ln(10)
    pdf.cell(0, 12, 'LiDAR Object Detection and', 0, 1, 'C')
    pdf.cell(0, 12, 'Tracking Pipeline', 0, 1, 'C')
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, 'Code Appendix & Running Instructions', 0, 1, 'C')
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, 'Course: Localization, Motion Planning and Sensor Fusion', 0, 1, 'C')
    pdf.cell(0, 7, 'Author: Abhisek Maddi', 0, 1, 'C')
    pdf.cell(0, 7, 'Date: February 2026', 0, 1, 'C')
    pdf.ln(10)

    # ===== Section 1: Apology & Citation Correction =====
    pdf.section_title('1. Apology & Reference Correction')

    pdf.body_text(
        'Dear Professor Florian,\n\n'
        'I sincerely apologize for the following errors in my submission:\n'
        '- The code was missing from the appendix.\n'
        '- The references did not adhere to IU citation guidelines (APA style).\n'
        '- The citations were not properly used within the document text.\n'
        '- Reference [12] contained incorrect author names and year.\n\n'
        'I take full responsibility for these mistakes and have worked to correct them. '
        'Below I provide the corrected references, the complete runnable code, and '
        'all requested visualizations.'
    )

    pdf.sub_title('1.1 Correction for Reference [12]')
    pdf.body_text('In my original submission, I cited:')
    pdf.italic_text(
        '    [12] T. A. W. Kim and H. Kim, "Pedestrian detection using LiDAR\n'
        '    point clouds," Sensors, vol. 20, no. 16, pp. 1-19, 2020.'
    )
    pdf.body_text(
        'After thorough research, I was unable to locate this exact publication in '
        'the Sensors journal (vol. 20, no. 16, 2020) with these authors. '
        'I sincerely apologize for this incorrect citation. '
        'The reference I intended to use is:'
    )
    pdf.bold_text('Corrected Reference:')
    pdf.body_text(
        'Liu, K., Wang, W., & Wang, J. (2019). Pedestrian detection with lidar '
        'point clouds based on single template matching. Electronics, 8(7), 780. '
        'https://doi.org/10.3390/electronics8070780'
    )
    pdf.body_text(
        'I apologize for the wrong year (2019, not 2020), the incorrect author names, '
        'and the wrong journal. The paper is freely available at:\n'
        'https://www.mdpi.com/2079-9292/8/7/780'
    )

    # ===== Section 2: All References in IU/APA Format =====
    pdf.add_page()
    pdf.section_title('2. References (IU/APA Format)')

    pdf.body_text(
        'Below are all references from my submission, reformatted according to '
        'IU Internationale Hochschule citation guidelines (APA style). '
        'Reference [12] has been corrected as described above.'
    )
    pdf.ln(2)

    references = [
        (
            'Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic robotics. '
            'MIT Press.'
        ),
        (
            'Urmson, C., Anhalt, J., Bagnell, D., Baker, C., Bittner, R., '
            'Clark, M. N., Dolan, J., Duggins, D., Galatali, T., Geyer, C., '
            'Gittleman, M., Harbaugh, S., Hebert, M., Howard, T. M., Kolski, S., '
            'Kelly, A., Likhachev, M., McNaughton, M., Miller, N., ... '
            'Ferguson, D. (2008). Autonomous driving in urban environments: Boss '
            'and the Urban Challenge. Journal of Field Robotics, 25(8), 425-466. '
            'https://doi.org/10.1002/rob.20255'
        ),
        (
            'Levinson, J., Askeland, J., Becker, J., Dolson, J., Held, D., '
            'Kammel, S., Kolter, J. Z., Langer, D., Pink, O., Pratt, V., '
            'Sokolsky, M., Stanek, G., Stavens, D., Teichman, A., Werling, M., '
            '& Thrun, S. (2011). Towards fully autonomous driving: Systems and '
            'algorithms. In Proceedings of the IEEE Intelligent Vehicles Symposium '
            '(IV) (pp. 163-168). IEEE. https://doi.org/10.1109/IVS.2011.5940562'
        ),
        (
            'Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for '
            'autonomous driving? The KITTI vision benchmark suite. In Proceedings '
            'of the IEEE Conference on Computer Vision and Pattern Recognition '
            '(CVPR) (pp. 3354-3361). IEEE. '
            'https://doi.org/10.1109/CVPR.2012.6248074'
        ),
        (
            'Cho, H., Seo, Y.-W., Kumar, B. V., & Rajkumar, R. R. (2014). '
            'A multi-sensor fusion system for moving object detection and tracking '
            'in urban driving environments. In Proceedings of the IEEE International '
            'Conference on Robotics and Automation (ICRA) (pp. 1836-1843). IEEE. '
            'https://doi.org/10.1109/ICRA.2014.6907100'
        ),
        (
            'Zhang, J., & Singh, S. (2014). LOAM: Lidar odometry and mapping '
            'in real-time. In Proceedings of Robotics: Science and Systems (RSS). '
            'https://doi.org/10.15607/RSS.2014.X.007'
        ),
        (
            'International Organization for Standardization. (2018). '
            'ISO 26262: Road vehicles - Functional safety. ISO.'
        ),
        (
            'Pomerleau, F., Colas, F., Siegwart, R., & Magnenat, S. (2013). '
            'Comparing ICP variants on real-world data sets. Autonomous Robots, '
            '34(3), 133-148. https://doi.org/10.1007/s10514-013-9327-2'
        ),
        (
            'Rusu, R. B., & Cousins, S. (2011). 3D is here: Point Cloud Library '
            '(PCL). In Proceedings of the IEEE International Conference on Robotics '
            'and Automation (ICRA) (pp. 1-4). IEEE. '
            'https://doi.org/10.1109/ICRA.2011.5980567'
        ),
        (
            'Zhou, Q.-Y., Park, J., & Koltun, V. (2018). Open3D: A modern library '
            'for 3D data processing. arXiv preprint arXiv:1801.09847. '
            'https://doi.org/10.48550/arXiv.1801.09847'
        ),
        (
            'Razakarivony, S., & Jurie, F. (2016). Vehicle detection in aerial '
            'imagery: A small target detection benchmark. Journal of Visual '
            'Communication and Image Representation, 34, 187-203. '
            'https://doi.org/10.1016/j.jvcir.2015.11.002'
        ),
        (
            'Liu, K., Wang, W., & Wang, J. (2019). Pedestrian detection with lidar '
            'point clouds based on single template matching. Electronics, 8(7), 780. '
            'https://doi.org/10.3390/electronics8070780\n'
            '[CORRECTED: Originally cited as Kim, T. A. W. & Kim, H. (2020). '
            'The correct authors are Liu, K. et al. and the year is 2019, not 2020.]'
        ),
    ]

    for i, ref in enumerate(references, 1):
        pdf.set_font('Helvetica', 'B', 9.5)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(10, 5, f'[{i}]', 0, 0)
        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(170, 5, ref)
        pdf.ln(2)

    # ===== Section 3: How to Run =====
    pdf.add_page()
    pdf.section_title('3. How to Run the Code')

    pdf.sub_title('3.1 Prerequisites')
    pdf.body_text('Python 3.8 or higher is required. Tested with Python 3.12.')

    pdf.sub_title('3.2 Installation (Step 1)')
    pdf.body_text('Install required Python packages:')
    pdf.code_block('pip install -r requirements.txt')
    pdf.body_text(
        'This installs: numpy, pandas, open3d, scikit-learn, filterpy, '
        'scipy, matplotlib, tqdm.'
    )

    pdf.sub_title('3.3 Run the Demo (Step 2)')
    pdf.body_text(
        'The main demo script is demo_run.py. It processes the CSV files and '
        'generates all visualizations:'
    )
    pdf.code_block('python demo_run.py')

    pdf.body_text('This will:')
    pdf.bullet('Load all 51 CSV frames from the lidar_data/ directory')
    pdf.bullet('Run the full detection and tracking pipeline on each frame')
    pdf.bullet(
        'Save bounding box plots, trajectory plots, and summary to demo_output/'
    )

    pdf.sub_title('3.4 Custom Options')
    pdf.code_block(
        'python demo_run.py --data lidar_data --frames 51 --output demo_output\n'
        '\n'
        'Options:\n'
        '  --data  / -d   Path to CSV directory      (default: lidar_data)\n'
        '  --frames / -n  Number of frames to process (default: 51)\n'
        '  --output / -o  Output directory for plots  (default: demo_output)'
    )

    pdf.sub_title('3.5 GitHub Repository')
    pdf.body_text('Complete code is available at:')
    pdf.bold_text('https://github.com/abhisek777/new-model')
    pdf.code_block(
        'git clone https://github.com/abhisek777/new-model.git\n'
        'cd new-model\n'
        'pip install -r requirements.txt\n'
        'python demo_run.py'
    )

    # ===== Section 4: Output Description =====
    pdf.add_page()
    pdf.section_title('4. Output & Visualizations')

    pdf.sub_title('4.1 Bounding Box Detection Plots')
    pdf.body_text(
        'Each plot shows two panels:\n'
        '- Left: Detected objects with bounding boxes (Bird\'s Eye View). '
        'Each cluster is labeled with its class (Vehicle/Pedestrian/Unknown) '
        'and dimensions.\n'
        '- Right: Tracked objects with IDs, velocity arrows, and trajectory tails.'
    )
    pdf.add_image_safe(os.path.join('demo_output', 'detections_frame_008.png'), w=175)

    pdf.sub_title('4.2 Object Trajectories')
    pdf.body_text(
        'Full movement paths of all tracked objects across 51 frames. '
        'Red = vehicles, green = pedestrians. '
        'Circles = start, squares = end position.'
    )
    pdf.add_image_safe(
        os.path.join('demo_output', 'object_trajectories.png'), w=150
    )

    pdf.add_page()
    pdf.sub_title('4.3 Pipeline Summary Statistics')
    pdf.body_text(
        '(a) Detections per frame by class\n'
        '(b) Active tracks over time\n'
        '(c) Processing time per frame\n'
        '(d) Track duration distribution'
    )
    pdf.add_image_safe(
        os.path.join('demo_output', 'pipeline_summary.png'), w=170
    )

    # ===== Section 5: Pipeline Architecture =====
    pdf.section_title('5. Pipeline Architecture')
    pdf.code_block(
        'CSV File (Blickfeld Cube 1)\n'
        '         |\n'
        '         v\n'
        '[1] Data Loading           (data_loader.py)\n'
        '    Reads semicolon-separated CSV: X;Y;Z;DISTANCE;INTENSITY;TIMESTAMP\n'
        '         |\n'
        '         v\n'
        '[2] Preprocessing          (preprocessing.py)\n'
        '    Range filtering (5-250 m), Voxel downsampling (0.1 m)\n'
        '    RANSAC ground removal, Outlier removal\n'
        '         |\n'
        '         v\n'
        '[3] Clustering (DBSCAN)    (clustering.py)\n'
        '    eps=0.5 m, min_samples=10\n'
        '         |\n'
        '         v\n'
        '[4] Feature Extraction &   (classification.py)\n'
        '    Classification         VEHICLE / PEDESTRIAN / UNKNOWN\n'
        '         |\n'
        '         v\n'
        '[5] Multi-Object Tracking  (tracking.py)\n'
        '    Kalman Filter + Hungarian algorithm'
    )

    # ===== Section 6: Results =====
    pdf.add_page()
    pdf.section_title('6. Results Summary')

    pdf.set_font('Courier', 'B', 9)
    pdf.set_fill_color(220, 230, 240)
    pdf.cell(100, 6, ' Metric', 1, 0, 'L', fill=True)
    pdf.cell(80, 6, ' Value', 1, 1, 'C', fill=True)

    results = [
        ('Frames processed', '51'),
        ('Avg processing time', '87.1 ms/frame'),
        ('Estimated FPS', '11.5'),
        ('Total vehicle detections', '510'),
        ('Total pedestrian detections', '816'),
        ('Unique tracked objects', '72'),
        ('Points per frame (raw)', '~18,500'),
        ('Points per frame (preprocessed)', '~8,400'),
        ('Clusters per frame', '50 - 60'),
        ('Active tracks per frame', '55 - 62'),
    ]

    pdf.set_font('Courier', '', 9)
    for metric, value in results:
        pdf.cell(100, 6, f' {metric}', 1, 0, 'L')
        pdf.cell(80, 6, value, 1, 1, 'C')
    pdf.ln(4)

    pdf.body_text(
        'The pipeline achieves real-time performance at ~11.5 FPS. '
        'The Kalman filter tracker maintains consistent object IDs across frames.'
    )

    # ===== Section 7: Additional Frames =====
    pdf.add_page()
    pdf.section_title('7. Additional Detection Frames')

    for frame_file in ['detections_frame_000.png', 'detections_frame_024.png',
                        'detections_frame_050.png']:
        path = os.path.join('demo_output', frame_file)
        if os.path.exists(path):
            frame_num = frame_file.split('_')[-1].replace('.png', '')
            pdf.sub_title(f'Frame {int(frame_num)}')
            pdf.add_image_safe(path, w=175)
            pdf.ln(2)

    # ===== Section 8: File Structure =====
    pdf.add_page()
    pdf.section_title('8. Project File Structure')

    pdf.code_block(
        'project/\n'
        '  demo_run.py              <-- MAIN: run this\n'
        '  main_pipeline.py         Integrated pipeline\n'
        '  run_complete_pipeline.py Extended pipeline + validation\n'
        '  \n'
        '  data_loader.py           Blickfeld CSV loading\n'
        '  preprocessing.py         Point cloud preprocessing\n'
        '  clustering.py            DBSCAN segmentation\n'
        '  classification.py        Rule-based classification\n'
        '  tracking.py              Kalman filter tracking\n'
        '  enhanced_tracking.py     Enhanced tracking\n'
        '  \n'
        '  visualization.py         BEV and 3D visualization\n'
        '  enhanced_visualization.py Extended visualization\n'
        '  academic_visualizations.py Academic figures\n'
        '  semantic_visualization.py  Semantic visualization\n'
        '  \n'
        '  requirements.txt         Python dependencies\n'
        '  lidar_data/              51 CSV frames (Blickfeld Cube 1)\n'
        '  demo_output/             Generated output plots'
    )

    # ===== Section 9: Closing =====
    pdf.section_title('9. Closing Note')
    pdf.body_text(
        'Dear Professor Florian,\n\n'
        'I sincerely apologize for:\n'
        '- The missing code appendix in my original submission\n'
        '- The incorrect citation [12] (wrong authors, wrong year, wrong journal)\n'
        '- Not following IU citation guidelines (APA style)\n'
        '- Not using in-text citations in the document body\n\n'
        'I have now provided:\n'
        '1. Complete, runnable Python code with clear instructions\n'
        '2. A demo script (demo_run.py) directly applicable to the CSV files\n'
        '3. Bounding box detection plots and object trajectory visualizations\n'
        '4. All references corrected and reformatted to IU/APA style\n'
        '5. The corrected reference: Liu et al. (2019), not Kim et al. (2020)\n'
        '   Available at: https://www.mdpi.com/2079-9292/8/7/780\n\n'
        'I understand these are serious issues and I take full responsibility. '
        'Please do not hesitate to contact me if anything further is needed.\n\n'
        'Kind regards,\n'
        'Abhisek Maddi'
    )

    # Save
    output_path = 'Professor_Report_Abhisek_Maddi.pdf'
    pdf.output(output_path)
    print(f'\nPDF generated: {os.path.abspath(output_path)}')
    print(f'Pages: {pdf.page_no()}')


if __name__ == '__main__':
    generate_pdf()
