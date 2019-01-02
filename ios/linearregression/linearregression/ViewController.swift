//
//  ViewController.swift
//  linearregression
//
//  Created by 黄敬嘉 on 2019/1/2.
//  Copyright © 2019 jamelouis. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var populationLabel: UILabel!
    @IBOutlet weak var profitLabel: UILabel!
    @IBOutlet weak var populationSlider: UISlider!
    
    private let populationProfits = populations_profits()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func sliderValueChanged(_ sender: UISlider) {
        let populations = Double(sender.value);
        let input = populations_profitsInput(populations: populations)
        
        guard let output = try? populationProfits.prediction(input: input) else {
            return
        }
        
        let profits = output.profits
        populationLabel.text = String(populations)
        profitLabel.text = String(profits)
    }
    
}

