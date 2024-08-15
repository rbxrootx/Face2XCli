const fs = require('fs').promises;
const path = require('path');
const { program } = require('commander');
const { imageToNdarray, multiUpscale, initializeONNX } = require('./utils');
const sharp = require('sharp');
const ora = require('ora');
const chalk = require('chalk');
const figlet = require('figlet');
const inquirer = require('inquirer');
const ndarray = require('ndarray');

const { SingleBar, Presets } = require('cli-progress');



async function processImage(inputPath, outputPath, upscaleFactor) {
    try {
        const inputBuffer = await fs.readFile(inputPath);
        const image = sharp(inputBuffer);
        const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });

        const imageArray = ndarray(new Uint8Array(data), [info.width, info.height, info.channels]);

        const upscaledArray = await multiUpscale(imageArray, upscaleFactor);

        const upscaledWidth = info.width * Math.pow(2, upscaleFactor);
        const upscaledHeight = info.height * Math.pow(2, upscaleFactor);

        await sharp(Buffer.from(upscaledArray.data), {
            raw: {
                width: upscaledWidth,
                height: upscaledHeight,
                channels: info.channels,
            },
        })
            .png()
            .toFile(outputPath);

        console.log(`Processed: ${inputPath} -> ${outputPath}`);
        return true;
    } catch (error) {
        console.error(`Error processing ${inputPath}:`, error);
        return false;
    }
}



async function processBulkImages(inputDir, outputDir, upscaleFactor) {
    try {
        await fs.mkdir(outputDir, { recursive: true });
        const files = await fs.readdir(inputDir);
        const pngFiles = files.filter(file => path.extname(file).toLowerCase() === '.png');

        console.log(chalk.cyan(`Found ${pngFiles.length} PNG files to process.`));

        const initSpinner = ora('Initializing ONNX runtime').start();
        await initializeONNX(() => { });
        initSpinner.succeed('ONNX runtime initialized');

        const progressBar = new SingleBar({
            format: 'Processing |' + chalk.cyan('{bar}') + '| {percentage}% || {value}/{total} images',
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            hideCursor: true
        }, Presets.shades_classic);

        progressBar.start(pngFiles.length, 0);

        let successCount = 0;
        for (const [index, file] of pngFiles.entries()) {
            const inputPath = path.join(inputDir, file);
            const outputPath = path.join(outputDir, `${path.parse(file).name}_${Math.pow(2, upscaleFactor)}x.png`);
            const success = await processImage(inputPath, outputPath, upscaleFactor);
            if (success) successCount++;
            progressBar.update(index + 1);
        }

        progressBar.stop();

        console.log(chalk.green(`\nProcessing complete! ${successCount}/${pngFiles.length} images upscaled successfully.`));
    } catch (error) {
        console.error(chalk.red('Error processing images:'), error);
    }
}


async function main() {
    console.log(chalk.magenta(figlet.textSync('PNG Upscaler', { horizontalLayout: 'full' })));

    program
        .version('1.0.0')
        .description('Bulk PNG Upscaler - Now with extra sexiness!')
        .parse(process.argv);

    const questions = [
        {
            type: 'input',
            name: 'inputDir',
            message: 'Enter the input directory containing PNG files:',
            validate: async (input) => {
                try {
                    const stats = await fs.stat(input);
                    return stats.isDirectory() ? true : 'Please enter a valid directory path';
                } catch (error) {
                    return 'Please enter a valid directory path';
                }
            }
        },
        {
            type: 'input',
            name: 'outputDir',
            message: 'Enter the output directory for upscaled PNG files:',
            default: (answers) => path.join(answers.inputDir, 'upscaled')
        },
        {
            type: 'list',
            name: 'upscaleFactor',
            message: 'Choose the upscale factor:',
            choices: [
                { name: '2x', value: 1 },
                { name: '4x', value: 2 },
                { name: '8x', value: 3 }
            ]
        }
    ];

    const answers = await inquirer.prompt(questions);

    console.log(chalk.yellow('\nStarting bulk upscaling process...'));
    await processBulkImages(answers.inputDir, answers.outputDir, answers.upscaleFactor);

    console.log(chalk.magenta('\nThank you for using the Super Sexy PNG Upscaler!'));
}

main().catch(error => console.error(chalk.red('An unexpected error occurred:'), error));